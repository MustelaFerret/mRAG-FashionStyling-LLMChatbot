from __future__ import annotations

import re
import time
from typing import Any, Dict, List

import numpy as np
from PIL import Image

from src.backend.core.config import settings
from src.backend.core.query_logger import append_log
from src.backend.core.utils import extract_article_id_from_text, normalize_article_id, normalize_text
from src.backend.retrieval.encoders import QueryEncoder
from src.backend.retrieval.llm import (
    INTENT_CHAT,
    INTENT_COMPOSITE,
    INTENT_GRAPH,
    INTENT_SIMILAR,
    INTENT_VARIANT,
    QwenMultimodalService,
)
from src.backend.retrieval.qdrant import QdrantStore
from src.backend.services.catalog import FashionCatalog
from src.backend.services.compat_index import CompatPairingIndex
from src.scripts.graph.outfit_slots import get_slot

INTENT_CONFIRM_OPTIONS = [INTENT_SIMILAR, INTENT_GRAPH, INTENT_VARIANT]
COMPOSITE_INTENT_MESSAGE = (
    "Sorry, I'm a little confused by your last request. "
    "Could you help me confirm your intent so I can assist you better?"
)
HARD_FILTER_KEYS = ("product_type", "colour_group", "graphical_appearance")
# generic "shoe" maps to the catch-all type "Other shoe"; expand it to closed-shoe types so a
# pairing/search for "shoe" returns sneakers/flats/pumps but NOT boots or sandals.
_SHOE_TYPES = {"other shoe", "sneakers", "flat shoe", "flat shoes", "ballerinas", "pumps", "heels", "wedge"}
INTENT_NEEDS_REFERENCE = "needs_reference"
NEEDS_REFERENCE_MESSAGE = (
    "I'd be happy to find that in another colour — which item do you mean? "
    "Upload a photo or pick one of the items I just showed you, and I'll match its style in the colour you want."
)
NO_RESULTS_MESSAGE = "I could not find any matching items in the current inventory."
NO_PAIRING_MESSAGE = (
    "Sorry, I don't have learned outfit pairings for this specific item yet, "
    "so I can't confidently suggest pieces to match it."
)

# Cross-category suggestion thresholds (cross-encoder relevance, NOT keyword rules). Measured score
# gap: when the requested type genuinely lacks what was asked for, the best in-type item scores
# deeply negative (~ -5), while a type that does carry it scores ~ +6..+8 -- so a low in-type score
# plus a clearly higher out-of-type score is the signal. (md/fix_log_nasa_variant_suggest.md)
_SUGGEST_RELEVANCE_FLOOR = 2.0   # best in-type result must rate below this (a poor match) to suggest
_SUGGEST_RELEVANCE_MARGIN = 2.0  # an out-of-type item must beat the best in-type by at least this


class FashionRAGService:
    def __init__(self, embedder: QueryEncoder, store: QdrantStore, llm: QwenMultimodalService, catalog: FashionCatalog, limit: int = 5, personalization=None, sessions=None):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.catalog = catalog
        self.limit = max(1, int(limit))
        # how many items reach the UI (show-more reveals these); the reply text is grounded on
        # only the first `gen_limit` so it stays concise even when more cards are shown.
        self.ui_limit = max(self.limit, int(getattr(settings, "ui_item_limit", 10)))
        self.gen_limit = max(1, int(getattr(settings, "gen_context_items", self.limit)))
        self.personalization = personalization
        self.sessions = sessions
        self._reranker = None  # cross-encoder, lazy-loaded when settings.use_reranker
        self.compat_index = self._load_compat_index()
        self.intent_confirm_options = list(INTENT_CONFIRM_OPTIONS)
        self.allowed_filters = self._build_allowed_filters()
        self._pt_canon, self._pt_emb = self._build_pt_index()

    def _load_compat_index(self):
        if not settings.compat_pairing_fallback:
            return None
        try:
            index = CompatPairingIndex(settings.compat_dir)
            return index if index.ready else None
        except Exception:
            return None

    def _build_pt_index(self):
        canon = list(self.allowed_filters.get("product_type", []))
        siglip = getattr(self.embedder, "siglip", None)
        if not canon or siglip is None:
            return canon, None
        try:
            return canon, siglip.encode_texts(canon, show_progress=False)
        except Exception:
            return canon, None

    def _resolve_product_type(self, value: str) -> tuple[str, str]:
        # Returns (canonical_product_type, source). exact/synonym/corpus -> confident
        # (hard filter); siglip -> fuzzy OOV (soft, retrieval-only).
        exact = self.catalog.canonical_product_type(value)
        if exact:
            return exact, "exact"
        corpus = self.catalog.corpus_product_type(value)
        if corpus:
            return corpus, "corpus"
        siglip = getattr(self.embedder, "siglip", None)
        if self._pt_emb is None or siglip is None or not str(value).strip():
            return "", ""
        try:
            vec = np.asarray(siglip.encode_text(str(value)), dtype=np.float32)
        except Exception:
            return "", ""
        sims = self._pt_emb @ vec
        idx = int(np.argmax(sims))
        if float(sims[idx]) >= settings.product_type_match_threshold:
            return self._pt_canon[idx], "siglip"
        return "", ""

    def _build_allowed_filters(self) -> Dict[str, List[str]]:
        return {
            "product_type": list(getattr(self.catalog, "valid_product_types", []) or []),
            "colour_group": list(getattr(self.catalog, "valid_colors", []) or []),
            "fit": list(getattr(self.catalog, "valid_fits", []) or []),
            "occasion": list(getattr(self.catalog, "valid_occasions", []) or []),
            "seasonality": list(getattr(self.catalog, "valid_seasonalities", []) or []),
            "graphical_appearance": list(getattr(self.catalog, "valid_graphical_appearances", []) or []),
            "gender": ["Men", "Women"],  # virtual field; applied as a post-filter, not a Qdrant key
        }

    def _validate_filter_value(
        self,
        key: str,
        value: str,
        dropped: List[str] | None = None,
        debug: List[Dict[str, Any]] | None = None,
        scope: str = "must_filters",
    ) -> str:
        value_str = str(value).strip()
        if not value_str:
            return ""
        allowed = self.allowed_filters.get(key, [])
        if not allowed:
            return ""
        if key == "product_type":
            matched, _ = self._resolve_product_type(value_str)
        else:
            value_lower = value_str.lower()
            matched = next((v for v in allowed if v.lower() == value_lower), "")
        if not matched and dropped is not None:
            dropped.append(f"{key}={value_str}")
        if debug is not None:
            debug.append(
                {
                    "scope": scope,
                    "key": key,
                    "input": value_str,
                    "matched": matched,
                    "allowed_count": len(allowed),
                    "allowed_has_value": bool(matched),
                }
            )
        return matched

    def _validate_filters(
        self,
        filters: Dict[str, str],
        dropped: List[str] | None = None,
        debug: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, str]:
        validated: Dict[str, str] = {}
        for key, value in (filters or {}).items():
            cleaned = self._validate_filter_value(
                key,
                str(value).strip(),
                dropped=dropped,
                debug=debug,
                scope="must_filters",
            )
            if cleaned:
                validated[key] = cleaned
        return validated

    def _validate_must_not(
        self,
        filters: Dict[str, List[str] | str],
        dropped: List[str] | None = None,
        debug: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, List[str]]:
        validated: Dict[str, List[str]] = {}
        for key, value in (filters or {}).items():
            if isinstance(value, str):
                values = [value]
            else:
                values = list(value or [])
            cleaned_values: List[str] = []
            for item in values:
                cleaned = self._validate_filter_value(
                    key,
                    str(item).strip(),
                    dropped=dropped,
                    debug=debug,
                    scope="must_not_filters",
                )
                if cleaned:
                    cleaned_values.append(cleaned)
            if cleaned_values:
                validated[key] = cleaned_values
        return validated

    def _format_context(self, points: List[Any]) -> str:
        context_parts: List[str] = []
        for point in points or []:
            item = getattr(point, "payload", {}) or {}
            name = str(item.get("prod_name", "") or item.get("product_type", "") or "")
            category = str(item.get("product_type", "") or "")
            colour = str(item.get("colour_group", ""))
            description = str(item.get("description", "")).strip()
            if len(description) > 160:
                description = description[:160].rsplit(" ", 1)[0] + "..."
            context_parts.append(f"- {name} — {colour} {category}. {description}".strip())
        return "\n".join(context_parts)

    def _build_image_url(self, article_id: str) -> str:
        aid = normalize_article_id(article_id)
        if not aid:
            return ""
        return f"/images/{aid[:3]}/{aid}.jpg"

    def _card_from_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        aid = str(payload.get("article_id", ""))
        return {
            "article_id": aid,
            "name": str(payload.get("prod_name", "") or payload.get("product_type", "") or ""),
            "product_type": str(payload.get("product_type", "")),
            "product_group": str(payload.get("product_group", "")),
            "colour_group": str(payload.get("colour_group", "")),
            "fit": str(payload.get("fit", "")),
            "occasion": str(payload.get("occasion", "")),
            "seasonality": str(payload.get("seasonality", "")),
            "description": str(payload.get("description", "")),
            "image_path": self._build_image_url(aid),
            "price": str(payload.get("price", "") or ""),
        }

    def _rerank_doc(self, point) -> str:
        """Structured doc text the cross-encoder scores against (kept in sync across rerank uses)."""
        pl = getattr(point, "payload", {}) or {}
        head = " ".join(
            str(pl.get(k, "") or "")
            for k in ("colour_group", "graphical_appearance", "dominant_material",
                      "product_type", "fit", "occasion", "seasonality")
            if pl.get(k))
        return f"{head}. {pl.get('description', '')}".strip()

    def _cross_category_suggestions(self, query, product_type, primary_points, text_dense,
                                    sparse_idx, sparse_val, must_not_filters):
        """Relevance-driven cross-category suggestion -- no keyword rules. If the cross-encoder rates
        the best in-type result a poor match for the request, yet rates an item of ANOTHER type
        clearly higher, the requested type cannot satisfy the request, so offer those other-type
        items ("no NASA shirt, but here are NASA tees"). The generation layer phrases the specifics
        from the query + suggestions. Returns suggestion cards (empty when no suggestion applies)."""
        rr = self._get_reranker()
        if rr is None or not query or not primary_points:
            return []
        in_score = max(rr.score(query, [self._rerank_doc(p) for p in primary_points[:5]]))
        if in_score >= _SUGGEST_RELEVANCE_FLOOR:
            return []  # the requested type already matches the request well
        relaxed = self.store.hybrid_search(
            text_dense=text_dense, image_dense=None, sparse_indices=sparse_idx,
            sparse_values=sparse_val, limit=30, must_filters=None,
            must_not_filters=must_not_filters) or []
        pt_low = str(product_type).strip().lower()
        cross = [p for p in relaxed
                 if str((getattr(p, "payload", {}) or {}).get("product_type", "")).strip().lower() != pt_low][:12]
        if not cross:
            return []
        cross_scores = rr.score(query, [self._rerank_doc(p) for p in cross])
        order = sorted(range(len(cross)), key=lambda i: cross_scores[i], reverse=True)
        if cross_scores[order[0]] - in_score <= _SUGGEST_RELEVANCE_MARGIN:
            return []  # nothing outside the requested type matches clearly better
        cards: List[Dict[str, Any]] = []
        seen_codes: set = set()
        for i in order:
            payload = getattr(cross[i], "payload", {}) or {}
            code = str(payload.get("product_code", "")) or str(payload.get("article_id", ""))
            if code in seen_codes:
                continue
            seen_codes.add(code)
            cards.append(self._card_from_payload(payload))
            if len(cards) >= 4:
                break
        return cards

    def _hard_filters(self, must_filters: Dict[str, str]) -> Dict[str, str]:
        return {k: v for k, v in (must_filters or {}).items() if k in HARD_FILTER_KEYS and str(v).strip()}

    def _hybrid_with_relaxation(
        self,
        text_dense,
        image_dense,
        sparse_idx,
        sparse_val,
        must_filters: Dict[str, str],
        must_not_filters: Dict[str, List[str]],
        limit: int,
        retrieval_path: List[str],
    ) -> List[Any]:
        hard = self._hard_filters(must_filters)
        # occasion is NOT a hard key (an item legitimately Unknown for it must not be excluded
        # outright), but for an occasion-defined query with no product_type ("loungewear for
        # sleeping", "gym gear") it is the ONLY structured signal -- without it the raw query
        # latches onto literal token matches ("sleeping" -> "sleeping mask"). Apply it as the
        # TIGHTEST, first-relaxed cascade tier, with the existing relaxation as the safety net.
        occ = str((must_filters or {}).get("occasion", "")).strip()
        cascade: List[tuple[str, Dict[str, str]]] = []
        if occ and occ.lower() != "unknown":
            cascade.append(("filter+occasion", {**hard, "occasion": occ}))
        if hard:
            cascade.append(("hard_filters", hard))
        pt_only = {k: v for k, v in hard.items() if k == "product_type"}
        if pt_only and pt_only != hard:
            cascade.append(("relax_to_product_type", pt_only))
        cascade.append(("relax_no_filters", {}))

        for label, flt in cascade:
            retrieval_path.append(label)
            points = self.store.hybrid_search(
                text_dense=text_dense,
                image_dense=image_dense,
                sparse_indices=sparse_idx,
                sparse_values=sparse_val,
                limit=limit,
                must_filters=flt or None,
                must_not_filters=must_not_filters,
            ) or []
            if points:
                return points
        return []

    def _get_reranker(self):
        if self._reranker is None and settings.use_reranker:
            from src.backend.retrieval.reranker import CrossEncoderReranker
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def _rerank_blend(self, query: str, points: List[Any], top_k: int) -> List[Any]:
        """Re-order the fused pool with the cross-encoder, PURE rerank order. With the rich
        structured doc text, pure rerank dominates the RRF blend on every gold-set metric
        (nDCG 0.9397 vs 0.9066 at depth 30, hit@1 0.933, hit@5 1.0 — md/refine_3.MD), so
        the earlier blend was dropped."""
        rr = self._get_reranker()
        if rr is None or not query or len(points) <= 1:
            return points[:top_k]
        # structured head (colour, pattern, material, type, fit, occasion, season): +4.7pp rerank
        # nDCG over colour+type alone (md/refine_2.MD); keep in sync with eval_rerank._doc_text
        docs = [self._rerank_doc(p) for p in points]
        scores = rr.score(query, docs)
        order = sorted(range(len(points)), key=lambda i: scores[i], reverse=True)
        return [points[i] for i in order[:top_k]]

    def _image_knn_points(self, ref_emb, exclude_id: str, must_filters: Dict[str, str], must_not_filters: Dict[str, List[str]]) -> List[Any]:
        raw = self.store.hybrid_search(
            text_dense=None,
            image_dense=ref_emb,
            sparse_indices=None,
            sparse_values=None,
            limit=self.ui_limit + 5,
            must_filters=must_filters or None,
            must_not_filters=must_not_filters,
        ) or []
        ex = normalize_article_id(exclude_id) if exclude_id else ""
        points = [
            p for p in raw
            if not ex or normalize_article_id(str((getattr(p, "payload", {}) or {}).get("article_id", ""))) != ex
        ]
        return points[: self.ui_limit]

    _FILTER_ALIASES = {
        "product_type": ("product_type", "product_type_name"),
        "colour_group": ("colour_group", "colour_group_name"),
    }

    @staticmethod
    def _gender_of_payload(payload: Dict[str, Any]) -> str:
        # gender isn't a clean field; derive it from index_name / section_name. Divided / Sport /
        # Mama and the like stay "" (unknown/unisex) so a gender filter only drops the OPPOSITE sex.
        idx = normalize_text(str(payload.get("index_name", "") or ""))
        sec = str(payload.get("section_name", "") or "")
        sec_first = normalize_text(sec.split()[0]) if sec.split() else ""
        sec_norm = normalize_text(sec)
        if idx == "menswear" or sec_first == "men":
            return "Men"
        if idx == "ladieswear" or "ladies" in idx or sec_first in ("womens", "ladies", "mama") or "women" in sec_norm:
            return "Women"
        return ""

    def _filter_gender(self, points: List[Any], gender: str) -> List[Any]:
        if not gender:
            return points
        kept = [p for p in points if self._gender_of_payload(getattr(p, "payload", {}) or {}) in (gender, "")]
        return kept

    def _payload_field(self, payload: Dict[str, Any], key: str) -> str:
        for alias in self._FILTER_ALIASES.get(key, (key,)):
            v = payload.get(alias)
            if v:
                return str(v)
        return str(payload.get(key, ""))

    def _filter_points(self, points: List[Any], must_filters: Dict[str, str], must_not_filters: Dict[str, List[str]]) -> List[Any]:
        # Exact (normalized) equality + key aliases, matching the Qdrant / in-memory filter
        # semantics. The previous 2-way substring match wrongly admitted e.g. "Dark Blue" and
        # "Light Blue" for a colour_group="Blue" filter (measured ~7700 spurious items on "Blue"),
        # so graph-pairing filters behaved more loosely than hybrid retrieval.
        filtered: List[Any] = []
        for point in points or []:
            payload = getattr(point, "payload", {}) or {}
            ok = True
            for key, value in (must_filters or {}).items():
                expected = normalize_text(str(value))
                if not expected:
                    continue
                if normalize_text(self._payload_field(payload, key)) != expected:
                    ok = False
                    break
            if not ok:
                continue
            excluded = False
            for key, values in (must_not_filters or {}).items():
                actual = normalize_text(self._payload_field(payload, key))
                if actual and any(actual == normalize_text(str(v)) for v in (values or [])):
                    excluded = True
                    break
            if not excluded:
                filtered.append(point)
        return filtered

    def _finalize_log(self, log_payload: Dict[str, Any], started_at: float, result_count: int) -> None:
        log_payload["result_count"] = result_count
        log_payload["graph_used"] = log_payload.get("intent_hint") == "graph_pairing" and bool(log_payload.get("graph_anchor_id"))
        log_payload["latency_ms"] = int((time.perf_counter() - started_at) * 1000)
        log_payload["timing_ms"]["total"] = log_payload["latency_ms"]
        append_log(settings.log_dir, log_payload)

    def _prepare_chat(
        self,
        query: str,
        image: Image.Image | None,
        session_id: str | None,
        request_id: str | None,
        started_at: float,
        confirmed_intent: str | None = None,
        customer_id: str | None = None,
        selected_anchor_id: str | None = None,
        no_anchor: bool = False,
    ) -> tuple[List[dict], str, Dict[str, Any], bool, Dict[str, Any] | None]:
        analysis_started = time.perf_counter()
        analysis = self.llm.analyze_user_query(query, vocab=self.allowed_filters)
        analysis_ms = int((time.perf_counter() - analysis_started) * 1000)
        search_query = str(analysis.get("search_query_en", "") or "").strip() or query
        search_query_raw = search_query
        intent_hint = str(analysis.get("intent_hint", "") or "").strip()
        analysis_debug = analysis.get("debug", {}) if isinstance(analysis.get("debug"), dict) else {}
        intent_rules = analysis_debug.get("intent_rules", {}) if isinstance(analysis_debug, dict) else {}
        llm_filters_raw = analysis_debug.get("llm_filters_raw", analysis.get("must_filters", {}))
        llm_must_not_raw = analysis_debug.get("llm_must_not_raw", analysis.get("must_not_filters", {}))
        dropped_filters: List[str] = []
        dropped_must_not: List[str] = []
        filter_validation_debug: List[Dict[str, Any]] = []
        must_not_validation_debug: List[Dict[str, Any]] = []
        must_filters = self._validate_filters(
            analysis.get("must_filters", {}),
            dropped=dropped_filters,
            debug=filter_validation_debug,
        )
        must_not_filters = self._validate_must_not(
            analysis.get("must_not_filters", {}),
            dropped=dropped_must_not,
            debug=must_not_validation_debug,
        )

        # OOV product_type resolved only by fuzzy SigLIP-nearest -> demote to soft:
        # drop from hard filters (it would over-constrain, e.g. parka->Coat misses the
        # real Jacket-labelled parkas) and let dense+sparse retrieval rank instead.
        soft_product_type = ""
        raw_pt = str((llm_filters_raw or {}).get("product_type", "") or "").strip()
        if must_filters.get("product_type") and raw_pt:
            _, pt_source = self._resolve_product_type(raw_pt)
            if pt_source == "siglip":
                soft_product_type = must_filters.pop("product_type", "")

        confirmed_value = str(confirmed_intent or "").strip().lower()
        if confirmed_value in self.intent_confirm_options:
            intent_hint = confirmed_value
            if isinstance(analysis_debug, dict):
                analysis_debug["intent_override"] = {
                    "source": "user_confirmed",
                    "value": confirmed_value,
                }

        session_state = self.sessions.get_or_create(session_id)[1] if self.sessions else None
        no_anchor = bool(no_anchor)
        if no_anchor and session_state is not None and self.sessions:
            self.sessions.clear_user_anchor(session_state)
        # An anchor the user provides THIS turn (a typed #id, or a card they clicked) is always
        # honored and becomes the sticky session anchor, so a later refinement ("no pattern",
        # "flat soles") keeps the SAME item rather than re-anchoring on a returned result.
        explicit_anchor = ""
        if not no_anchor:
            explicit_anchor = (
                normalize_article_id(extract_article_id_from_text(query) or "")
                or (normalize_article_id(selected_anchor_id) if selected_anchor_id else "")
            )
            if explicit_anchor and session_state is not None and self.sessions:
                self.sessions.set_user_anchor(session_state, explicit_anchor)
        # the sticky anchor persists across turns until the user changes or clears it (no_anchor)
        session_anchor = "" if no_anchor else (normalize_article_id(session_state.user_anchor_id) if session_state else "")
        classifier_intent = str(((analysis_debug or {}).get("intent_classifier") or {}).get("intent", "")).strip()
        requested_colour = str(must_filters.get("colour_group", "")).strip()
        # a genuine colour-variant request ("any different colour of this one") carries a colour cue
        # plus a variant word; it differs from a terse negative refinement ("no, nothing in orange")
        # that only happens to classify as variant. The former must keep its variant route, not be
        # swept into refine-continuation below.
        is_variant_request = isinstance(intent_rules, dict) and bool(intent_rules.get("is_color_variant_request"))
        referential_anchor = explicit_anchor or session_anchor
        # The text-only classifier cannot see the sticky/selected anchor. When it predicts a colour
        # variant and an anchor (or image) is actually present, pin the variant sub-path by what the
        # user mentions: a target colour -> that-colour branch; only "other/different colours"
        # without a target -> the same-type-other-colours branch.
        if classifier_intent == INTENT_VARIANT and (image is not None or referential_anchor):
            _qn = normalize_text(query)
            mentions_colour = "colour" in _qn or "color" in _qn
            if requested_colour:
                intent_hint = INTENT_VARIANT
                if isinstance(intent_rules, dict):
                    intent_rules["rule_applied"] = "honor_color_variant_with_anchor"
            elif mentions_colour:
                intent_hint = INTENT_VARIANT
                if isinstance(intent_rules, dict):
                    intent_rules["rule_applied"] = "honor_other_colours_with_anchor"

        # Symmetric to the variant re-honor: a confident graph_pairing prediction can be demoted by
        # the text-only rule layer when the query references the anchor implicitly ("what goes well
        # with it") -- no anchor WORD and no hardcoded pairing phrase. With a real clicked/session
        # anchor the referent demonstrably exists, so re-honor pairing. Guard against the classifier's
        # graph false-positives on self-contained searches ("something to wear to the gym" -> graph):
        # only re-honor when the query did not extract its own defining product_type/occasion.
        if (classifier_intent == INTENT_GRAPH and referential_anchor
                and intent_hint == INTENT_SIMILAR
                and not (must_filters.get("product_type") or must_filters.get("occasion"))):
            intent_hint = INTENT_GRAPH
            if isinstance(intent_rules, dict):
                intent_rules["rule_applied"] = "honor_graph_pairing_with_anchor"

        # Refinement continuation: a terse follow-up that carries a fashion constraint ("without a
        # pattern", "flat soles", "in beige") often reads as chit-chat/ambiguous in isolation. With a
        # live anchor (or image) and a retrieval intent on the previous turn, continue that SAME
        # intent on the SAME anchor instead of dropping to small-talk. Guarded by an extracted
        # attribute (must/must_not), so plain greetings ("thanks!") still fall through to chit-chat.
        last_intent = (session_state.last_intent if session_state else "") or ""
        last_pt = (session_state.last_product_type if session_state else "") or ""
        # A terse refinement reads as chit-chat/ambiguous in isolation, OR as a colour-variant with
        # NO target colour ("no, nothing in orange" -> classifier says variant, but there is no
        # colour to vary toward). Both are really "keep doing the last thing, with this constraint".
        refine_candidate = intent_hint in (INTENT_CHAT, INTENT_COMPOSITE) or (
            intent_hint == INTENT_VARIANT and not requested_colour and not is_variant_request)
        if (last_intent in (INTENT_SIMILAR, INTENT_GRAPH, INTENT_VARIANT)
                and refine_candidate
                and (must_filters or must_not_filters)
                and (referential_anchor or image is not None or last_pt)
                and not confirmed_value):
            cont = last_intent
            # a colour-NEGATIVE refinement is not a colour variant -> continue the similar search
            # excluding that colour, rather than the variant path that needs a target colour/anchor.
            if cont == INTENT_VARIANT and not requested_colour:
                cont = INTENT_SIMILAR
            intent_hint = cont
            if isinstance(intent_rules, dict):
                intent_rules["rule_applied"] = "refine_continue_last_intent"

        # Target-type inheritance (independent of the intent rule above, because the classifier is
        # noisy and may even keep a retrieval intent while dropping the noun): a constrained follow-up
        # that names NO garment ("nothing in orange", "without pattern", "in white") keeps searching
        # the previous category instead of collapsing to a generic pool. Pairing is excluded (its
        # product_type is the pairing TARGET, handled separately).
        if (last_pt and not must_filters.get("product_type")
                and (must_filters or must_not_filters)
                and intent_hint in (INTENT_SIMILAR, INTENT_VARIANT)
                and not confirmed_value):
            must_filters["product_type"] = last_pt
            if isinstance(intent_rules, dict):
                intent_rules["rule_applied"] = (intent_rules.get("rule_applied", "") + "+inherit_product_type").lstrip("+")

        # an uploaded image is a strong reference signal: never route it to small-talk
        if image is not None and intent_hint in (INTENT_CHAT, INTENT_COMPOSITE):
            intent_hint = INTENT_SIMILAR
            if isinstance(intent_rules, dict):
                intent_rules["rule_applied"] = "image_implies_similar"

        retrieval_path: List[str] = []
        direct_response: Dict[str, Any] | None = None
        if intent_hint == INTENT_COMPOSITE:
            retrieval_path.append("intent_composite")
            direct_response = {
                "type": INTENT_COMPOSITE,
                "message": COMPOSITE_INTENT_MESSAGE,
                "intent_options": self.intent_confirm_options,
                "intent_query": query,
            }
        elif intent_hint == INTENT_CHAT:
            retrieval_path.append("intent_chit_chat")
            direct_response = {"type": INTENT_CHAT}
        elif intent_hint == INTENT_VARIANT and not (image is not None or referential_anchor):
            retrieval_path.append("variant_needs_reference")
            direct_response = {"type": INTENT_NEEDS_REFERENCE, "message": NEEDS_REFERENCE_MESSAGE}

        if direct_response is not None:
            log_payload = {
                "event": "chat",
                "request_id": request_id or "",
                "session_id": session_id or "",
                "query": query,
                "search_query": search_query,
                "search_query_raw": search_query_raw,
                "dense_query": search_query,
                "sparse_query": search_query,
                "boosted_product_type": "",
                "boost_applied": False,
                "intent_hint": intent_hint,
                "analysis_debug": analysis_debug,
                "llm_filters_raw": llm_filters_raw,
                "llm_must_not_raw": llm_must_not_raw,
                "must_filters": must_filters,
                "must_not_filters": must_not_filters,
                "dropped_filters": dropped_filters,
                "dropped_must_not": dropped_must_not,
                "filter_validation_debug": filter_validation_debug,
                "must_not_validation_debug": must_not_validation_debug,
                "retrieval_path": retrieval_path,
                "active_anchor_id": referential_anchor,
                "no_anchor": no_anchor,
                "graph_anchor_id": "",
                "graph_anchor_source": "",
                "graph_neighbor_ids": [],
                "graph_candidate_count": 0,
                "graph_filtered_count": 0,
                "hybrid_result_count": 0,
                "has_image": bool(image),
                "result_ids": [],
                "timing_ms": {
                    "analysis": analysis_ms,
                    "embedding": 0,
                },
            }
            return [], "", log_payload, False, direct_response

        # gender is a VIRTUAL filter (no clean Qdrant field) -> pull it out of must/must_not so it
        # never reaches the hard filter or _filter_points (which would drop every item), and apply
        # it as a post-filter on the retrieved points instead.
        gender_filter = str(must_filters.pop("gender", "")).strip()
        (must_not_filters or {}).pop("gender", None)

        dense_query = search_query
        sparse_query = search_query
        product_type = str(must_filters.get("product_type", "")).strip()
        boost_applied = False
        if product_type:
            # BM25 encode_query dedups tokens into a set with binary weights, so repeating
            # the term was a no-op; the only real effect is ensuring the PT tokens are present.
            sparse_query = f"{product_type} {search_query}".strip()
            boost_applied = True

        embed_started = time.perf_counter()
        q = self.embedder.encode(text=dense_query, image=image, sparse_text=sparse_query)
        text_dense = q.get("text_dense")
        image_dense = q.get("image_dense")
        sparse_idx = q.get("sparse_indices", [])
        sparse_val = q.get("sparse_values", [])
        embed_ms = int((time.perf_counter() - embed_started) * 1000)
        points: List[Any] = []
        anchor_id = ""
        graph_anchor_source = ""
        graph_neighbor_ids: List[str] = []
        graph_candidate_count = 0
        graph_filtered_count = 0
        graph_anchor_invalid = ""
        hybrid_result_count = 0
        # NOTE: retrieval_path was initialized before the direct-response block; every
        # branch that appends to it also sets direct_response and returns early, so it is
        # guaranteed empty here (the old re-init was redundant and would have masked
        # markers if a non-returning branch were ever added).

        if intent_hint == "graph_pairing":
            retrieval_path.append("intent_graph_pairing")
            anchor_id = normalize_article_id(extract_article_id_from_text(query) or "")
            if anchor_id:
                graph_anchor_source = "query_reference"
            if not anchor_id and selected_anchor_id and not no_anchor:
                anchor_id = normalize_article_id(selected_anchor_id)
                if anchor_id:
                    graph_anchor_source = "selected_anchor"
                    retrieval_path.append("anchor_from_selection")
            if not anchor_id and session_anchor:
                # sticky session anchor: a follow-up pairing refinement ("no pattern") keeps the
                # ORIGINAL item the user paired with, instead of re-deriving a new one from the
                # refined query text (which would re-anchor onto a returned result).
                anchor_id = session_anchor
                graph_anchor_source = "session_anchor"
                retrieval_path.append("anchor_from_session")
            if not anchor_id:
                retrieval_path.append("anchor_from_hybrid_top1")
                # in "X to go with Y" the extracted product_type is the TARGET (X), while
                # the anchor to search for is Y. Constraining this search by the target's
                # type guaranteed anchor==target category ("what trousers go with a navy
                # blazer" anchored on trousers and returned blazers). So: drop product_type
                # from must and exclude the target type explicitly.
                target_pt = str(must_filters.get("product_type", "") or soft_product_type).strip()
                target_slot = get_slot(target_pt) if target_pt else ""
                if target_slot in ("", "other") and target_pt:
                    target_slot = self.catalog.infer_slot_from_text(target_pt)
                anchor_must = {k: v for k, v in must_filters.items() if k != "product_type"}
                # the anchor is described by the query MINUS the target mention ("black
                # leather jacket bag" -> "black leather jacket"); searching with the target
                # words in floods the pool with target-category items (measured: top-10 all
                # Bag for the bag query). Strip target tokens and re-encode for this search.
                anchor_query = search_query
                if target_pt:
                    target_words = {w for w in re.findall(r"[a-z0-9]+", target_pt.lower())}
                    kept = [w for w in anchor_query.split() if w.lower().strip(".,!?") not in target_words]
                    if kept:
                        anchor_query = " ".join(kept)
                aq = self.embedder.encode(text=anchor_query, image=None, sparse_text=anchor_query)
                anchor_points = self.store.hybrid_search(
                    text_dense=aq.get("text_dense") or text_dense,
                    image_dense=image_dense,
                    sparse_indices=aq.get("sparse_indices", []),
                    sparse_values=aq.get("sparse_values", []),
                    limit=10,
                    must_filters=self._hard_filters(anchor_must) or None,
                    must_not_filters=must_not_filters,
                )
                # PT-exact exclusion is too narrow (target "shoes" would not exclude an
                # "Other shoe" anchor): pick the first candidate whose SLOT differs from
                # the target slot. If every candidate is target-slot, drop the anchor —
                # a generic result beats a slot-inverted pairing.
                first = None
                for cand in anchor_points or []:
                    cand_id = str((getattr(cand, "payload", {}) or {}).get("article_id", ""))
                    if target_slot and self.catalog.infer_article_slot(cand_id) == target_slot:
                        continue
                    first = cand
                    break
                anchor_id = str((getattr(first, "payload", {}) or {}).get("article_id", "")) if first else ""
                if anchor_id:
                    graph_anchor_source = "hybrid_top1"

            anchor_id = normalize_article_id(anchor_id)
            if anchor_id and not self.catalog.get_meta(anchor_id):
                # unknown anchor (malformed id / stale session): without catalog meta the
                # slot logic below cannot constrain the target -> drop it instead of
                # proceeding unconstrained, and leave a trace in the log payload.
                graph_anchor_invalid = anchor_id
                anchor_id = ""
            anchor_slot = get_slot(self.catalog.get_meta(anchor_id).get("product_type_name", "")) if anchor_id else ""

            # Requested target category (e.g. "shoes"): keep its SLOT as a constraint even
            # when product_type was demoted to soft. In "X to match [the Y]" the LLM often
            # grabs the anchor's own type (Y) instead of the target (X); a pairing target must
            # differ from the anchor slot, so if it collides we discard it and re-infer the
            # target from the query text, excluding the anchor slot.
            requested_pt = str(must_filters.get("product_type", "") or soft_product_type).strip()
            requested_slot = get_slot(requested_pt) if requested_pt else ""
            if requested_slot == "other":
                requested_slot = ""
            if anchor_slot and requested_slot == anchor_slot:
                must_filters.pop("product_type", None)
                requested_slot = self.catalog.infer_slot_from_text(query, exclude_slot=anchor_slot)
            elif anchor_slot and not requested_slot:
                requested_slot = self.catalog.infer_slot_from_text(query, exclude_slot=anchor_slot)

            # Pairing target is category-level: constrain by SLOT (requested_slot), not the
            # specific product_type. A generic word like "shoe" resolves to a narrow type
            # ("Other shoe") that would wrongly exclude Sneakers/Boots; the slot filter keeps
            # all footwear so any matching shoe (incl. a blue sneaker) can surface.
            pairing_must = {k: v for k, v in must_filters.items() if k != "product_type"}

            if anchor_id:
                retrieval_path.append("graph_neighbors")
                neighbor_ids = self.catalog.get_graph_diverse_neighbors(
                    anchor_id,
                    limit=settings.graph_pair_limit,
                    max_per_pt=settings.graph_max_per_pt,
                    preferred_min_weight=settings.graph_preferred_min_weight,
                    hard_min_weight=settings.graph_hard_min_weight,
                )
                graph_neighbor_ids = list(neighbor_ids or [])
                graph_candidate_count = len(graph_neighbor_ids)
                points = self.store.retrieve_by_article_ids(graph_neighbor_ids)
                points = self._filter_points(points, pairing_must, must_not_filters)
                if requested_slot:
                    points = [p for p in points if self.catalog.infer_article_slot(str((getattr(p, "payload", {}) or {}).get("article_id", ""))) == requested_slot]
                graph_filtered_count = len(points)

                if not points and graph_candidate_count == 0 and self.compat_index is not None:
                    # COLD tier 1 — fuse the two cold methods that EACH match the warm co-buy quality
                    # ceiling (VLM-judged outfit-compat 4.62 == warm; P3alpha alone only 3.88). RRF of
                    # visual-twin borrow + compatibility embedding beat either alone on held-out co-buy
                    # recall@10 (0.55/0.57 -> 0.64) AND on outfit quality (-> 4.67); compat covers 100%
                    # of the catalog so this loses no reach (md/exp_pairing_coldtier.md). Gated to a
                    # truly cold anchor (no co-buys at all): "has co-buys but none match the target"
                    # still returns no pairing rather than a fabricated visual guess.
                    twins = self.compat_index.nearest_warm_twins(anchor_id, self.catalog.graph_adj, k=5)
                    borrowed: Dict[str, float] = {}
                    for twin_id, sim in twins:
                        for nid in self.catalog.get_graph_diverse_neighbors(
                            twin_id,
                            limit=settings.graph_pair_limit * 2,
                            max_per_pt=settings.graph_max_per_pt,
                            preferred_min_weight=settings.graph_preferred_min_weight,
                            hard_min_weight=settings.graph_hard_min_weight,
                        ):
                            nid = normalize_article_id(nid)
                            borrowed[nid] = borrowed.get(nid, 0.0) + sim
                    twin_rank = sorted(borrowed, key=lambda n: borrowed[n], reverse=True)
                    compat_rank = [normalize_article_id(c) for c in self.compat_index.complement_ids(
                        anchor_id, self.limit * 10, target_slot=requested_slot)]
                    rrf_score: Dict[str, float] = {}
                    for ranked in (twin_rank, compat_rank):
                        for r, nid in enumerate(ranked):
                            rrf_score[nid] = rrf_score.get(nid, 0.0) + 1.0 / (60 + r)
                    fused = sorted(rrf_score, key=lambda n: rrf_score[n], reverse=True)
                    if fused:
                        fpoints = self.store.retrieve_by_article_ids(fused)
                        fpoints = self._filter_points(fpoints, pairing_must, must_not_filters)
                        if requested_slot:
                            fpoints = [p for p in fpoints if self.catalog.infer_article_slot(str((getattr(p, "payload", {}) or {}).get("article_id", ""))) == requested_slot]
                        frank = {normalize_article_id(o): i for i, o in enumerate(fused)}
                        fpoints.sort(key=lambda p: frank.get(
                            normalize_article_id(str((getattr(p, "payload", {}) or {}).get("article_id", ""))), 1_000_000))
                        if fpoints:
                            retrieval_path.append("cold_twin_compat_rrf")
                            points = fpoints[: settings.graph_pair_limit]
                            graph_neighbor_ids = [str((getattr(p, "payload", {}) or {}).get("article_id", "")) for p in points]

                if not points and self.catalog.aux_adj:
                    # COLD supplement — P3alpha transaction edges (the co-buy COMPLEMENT, so it targets
                    # future-basket partners, not the current outfit; VLM-judged 3.88 < twin/compat
                    # 4.62, so it now runs only AFTER the RRF tier, as a backstop / for the
                    # graph_candidate_count>0-but-filtered-empty case). md/refine_5.MD, exp_pairing_coldtier.md.
                    aux_ids = self.catalog.get_graph_diverse_neighbors(
                        anchor_id,
                        limit=settings.graph_pair_limit * 2,
                        max_per_pt=settings.graph_max_per_pt,
                        preferred_min_weight=settings.graph_preferred_min_weight,
                        hard_min_weight=settings.graph_hard_min_weight,
                        use_aux=True,
                    )
                    if aux_ids:
                        apoints = self.store.retrieve_by_article_ids(aux_ids)
                        apoints = self._filter_points(apoints, pairing_must, must_not_filters)
                        if requested_slot:
                            apoints = [p for p in apoints if self.catalog.infer_article_slot(str((getattr(p, "payload", {}) or {}).get("article_id", ""))) == requested_slot]
                        if apoints:
                            retrieval_path.append("p3a_cold_neighbors")
                            points = apoints[: settings.graph_pair_limit]
                            graph_neighbor_ids = [
                                str((getattr(p, "payload", {}) or {}).get("article_id", "")) for p in points
                            ]
                if not points and self.compat_index is not None and graph_candidate_count == 0:
                    # COLD tier 2 — visual-compatibility GUESS. Only used when the anchor is truly
                    # cold (no co-buy neighbours at all). If it HAS co-buys but none match the target
                    # (graph_candidate_count > 0, filtered to 0), the data says people don't buy this
                    # target with it -> trust that and return no pairing, rather than fabricating a
                    # visual guess (the "boots that don't match the trousers" hallucination).
                    twins = self.compat_index.nearest_warm_twins(anchor_id, self.catalog.graph_adj, k=5)
                    if twins:
                        borrowed: Dict[str, float] = {}
                        for twin_id, sim in twins:
                            for nid in self.catalog.get_graph_diverse_neighbors(
                                twin_id,
                                limit=settings.graph_pair_limit * 2,
                                max_per_pt=settings.graph_max_per_pt,
                                preferred_min_weight=settings.graph_preferred_min_weight,
                                hard_min_weight=settings.graph_hard_min_weight,
                            ):
                                borrowed[nid] = borrowed.get(nid, 0.0) + sim
                        if borrowed:
                            order = sorted(borrowed, key=lambda n: borrowed[n], reverse=True)
                            bpoints = self.store.retrieve_by_article_ids(order)
                            bpoints = self._filter_points(bpoints, pairing_must, must_not_filters)
                            if requested_slot:
                                bpoints = [p for p in bpoints if self.catalog.infer_article_slot(str((getattr(p, "payload", {}) or {}).get("article_id", ""))) == requested_slot]
                            brank = {normalize_article_id(o): i for i, o in enumerate(order)}
                            bpoints.sort(key=lambda p: brank.get(
                                normalize_article_id(str((getattr(p, "payload", {}) or {}).get("article_id", ""))), 1_000_000))
                            if bpoints:
                                retrieval_path.append("borrowed_neighbors")
                                points = bpoints[: settings.graph_pair_limit]
                                graph_neighbor_ids = [
                                    str((getattr(p, "payload", {}) or {}).get("article_id", "")) for p in points
                                ]
                if not points and self.compat_index is not None:
                    cand_ids = self.compat_index.complement_ids(anchor_id, self.limit * 10, target_slot=requested_slot)
                    if cand_ids:
                        retrieval_path.append("compat_pairing_fallback")
                        rank = {normalize_article_id(c): i for i, c in enumerate(cand_ids)}
                        cpoints = self.store.retrieve_by_article_ids(cand_ids)
                        cpoints = self._filter_points(cpoints, pairing_must, must_not_filters)
                        cpoints.sort(key=lambda p: rank.get(
                            normalize_article_id(str((getattr(p, "payload", {}) or {}).get("article_id", ""))), 1_000_000))
                        points = cpoints[:settings.graph_pair_limit]
                        graph_neighbor_ids = [str((getattr(p, "payload", {}) or {}).get("article_id", "")) for p in points]

        elif intent_hint == INTENT_SIMILAR:
            ref_emb = None
            ref_id = ""
            ref_type = ""
            ref_source = ""
            if image is not None and image_dense:
                ref_emb = image_dense
                ref_source = "uploaded_image"
            else:
                rid = extract_article_id_from_text(query)
                if rid:
                    ref_id = normalize_article_id(rid)
                    ref_source = "query_reference"
                elif referential_anchor:
                    ref_id = referential_anchor
                    ref_source = "selected_anchor" if explicit_anchor else "session_anchor"
                if ref_id:
                    ref_emb = self.store.get_named_vector(ref_id, settings.vector_name_image)
                    ref_type = self.catalog.get_meta(ref_id).get("product_type_name", "")
            if ref_emb:
                retrieval_path.append("similar_image_knn")
                anchor_id = ref_id
                graph_anchor_source = ref_source
                sim_must = self._hard_filters(must_filters)
                if "product_type" not in sim_must and ref_type:
                    sim_must["product_type"] = ref_type
                points = self._image_knn_points(ref_emb, ref_id, sim_must, must_not_filters)
                hybrid_result_count = len(points)

        elif intent_hint == INTENT_VARIANT:
            ref_emb = None
            ref_id = ""
            ref_type = ""
            ref_source = ""
            if image is not None and image_dense:
                ref_emb = image_dense
                ref_source = "uploaded_image"
            elif referential_anchor:
                ref_id = referential_anchor
                ref_source = "query_reference" if extract_article_id_from_text(query) else ("selected_anchor" if explicit_anchor else "session_anchor")
                ref_emb = self.store.get_named_vector(ref_id, settings.vector_name_image)
                ref_type = self.catalog.get_meta(ref_id).get("product_type_name", "")
            if ref_emb and requested_colour:
                retrieval_path.append("color_variant_image_knn")
                anchor_id = ref_id
                graph_anchor_source = ref_source
                var_must = {"colour_group": requested_colour}
                if ref_type:
                    var_must["product_type"] = ref_type
                if must_filters.get("graphical_appearance"):
                    var_must["graphical_appearance"] = must_filters["graphical_appearance"]
                points = self._image_knn_points(ref_emb, ref_id, var_must, must_not_filters)
                hybrid_result_count = len(points)
            elif ref_emb:
                # "in a different / other colour" with NO specific target colour: show the SAME item
                # type in OTHER colours -> image-KNN of the anchor's type, excluding its own colour
                # (previously this fell through to a generic text search and returned unrelated junk).
                retrieval_path.append("color_variant_other_colours")
                anchor_id = ref_id
                graph_anchor_source = ref_source
                oc_must = {"product_type": ref_type} if ref_type else {}
                anchor_colour = self.catalog.get_meta(ref_id).get("colour_group_name", "")
                oc_must_not = dict(must_not_filters or {})
                if anchor_colour:
                    oc_must_not["colour_group"] = list({*(oc_must_not.get("colour_group") or []), anchor_colour})
                points = self._image_knn_points(ref_emb, ref_id, oc_must, oc_must_not)
                hybrid_result_count = len(points)

        if intent_hint == "graph_pairing" and points and requested_pt:
            # the user named a SPECIFIC target type ("trousers", "shoe") -> keep ONLY that exact
            # product type, so a Skirt/Shorts cannot pass for "trousers" and a Boot cannot pass for
            # "shoe". Quality over quantity: if nothing of that type pairs, return none (-> the
            # honest "no pairings" message) rather than padding with off-type items from the slot.
            _rpt = normalize_text(self.catalog.canonical_product_type(requested_pt) or requested_pt)
            # generic "shoe" -> "Other shoe": accept the whole closed-shoe family (sneakers/flats/
            # pumps), excluding boots/sandals; a concrete type ("Boots", "Trousers") stays exact.
            _allowed = _SHOE_TYPES if _rpt == "other shoe" else {_rpt}

            def _ptype(p):
                aid = normalize_article_id(str((getattr(p, "payload", {}) or {}).get("article_id", "")))
                return normalize_text(self.catalog.get_meta(aid).get("product_type_name", ""))

            exact = [p for p in points if _ptype(p) in _allowed]
            points = exact
            if exact:
                retrieval_path.append("pairing_exact_type")
            else:
                retrieval_path.append("pairing_no_exact_type")

        if intent_hint == "graph_pairing" and len(points) > 1 and settings.use_reranker and query:
            # the candidates are confident co-buy / P3a pairings already; reorder THEM by relevance
            # to the QUERY so free-text nuance ("elegant", "minimal", a material) the structured
            # filters can't capture still steers the result -- staying close to the query instead of
            # only intent + slot. (Does not add items, so it cannot reintroduce off-pairing junk.)
            points = self._rerank_blend(query, points, self.ui_limit)
            retrieval_path.append("pairing_query_rerank")

        if not points and intent_hint == "graph_pairing":
            retrieval_path.append("no_graph_pairing")
        elif not points:
            do_rerank = settings.use_reranker and bool(query)
            pool = max(settings.rerank_candidate_depth, self.ui_limit) if do_rerank else self.ui_limit
            points = self._hybrid_with_relaxation(
                text_dense=text_dense,
                image_dense=image_dense,
                sparse_idx=sparse_idx,
                sparse_val=sparse_val,
                must_filters=must_filters,
                must_not_filters=must_not_filters,
                limit=pool,
                retrieval_path=retrieval_path,
            )
            if do_rerank and len(points) > 1:
                points = self._rerank_blend(query, points, self.ui_limit)
                retrieval_path.append("cross_encoder_rerank")
            else:
                points = points[:self.ui_limit]
            hybrid_result_count = len(points)

        # gender post-filter: drop the opposite sex (keep matching + unisex). Skip if it would
        # empty the results -- a partial gender hint should not return nothing.
        if gender_filter and points:
            gendered = self._filter_gender(points, gender_filter)
            if gendered:
                points = gendered
                retrieval_path.append("gender_filter:" + gender_filter)

        items: List[dict] = []
        for point in points:
            items.append(self._card_from_payload(getattr(point, "payload", {}) or {}))

        # Cross-category suggestion: an honest "no good <type> for this, but here is a better match
        # elsewhere". Only for a plain typed search (similar intent + a product_type filter); the
        # cross-encoder relevance decides whether the requested type actually fails the request.
        suggestion_cards: List[dict] = []
        if intent_hint == INTENT_SIMILAR and must_filters.get("product_type") and points:
            suggestion_cards = self._cross_category_suggestions(
                query, str(must_filters.get("product_type", "")), points,
                text_dense, sparse_idx, sparse_val, must_not_filters)
            if suggestion_cards:
                retrieval_path.append("cross_category_suggestion")

        if self.personalization is not None and customer_id and self.personalization.has_profile(customer_id):
            order = self.personalization.rerank(customer_id, [it.get("article_id", "") for it in items])
            pos = {aid: i for i, aid in enumerate(order)}
            items.sort(key=lambda it: pos.get(it.get("article_id", ""), len(items)))
            retrieval_path.append("personalized_rerank")

        result_ids = [item.get("article_id", "") for item in items if item.get("article_id")]
        if session_state is not None:
            # remember the retrieval intent + target type so a terse follow-up can continue them
            if intent_hint in (INTENT_SIMILAR, INTENT_GRAPH, INTENT_VARIANT):
                session_state.last_intent = intent_hint
                _pt = str(must_filters.get("product_type", "") or "").strip()
                if _pt:
                    session_state.last_product_type = _pt
            if result_ids:
                # cache results for reference only; the sticky user anchor is left untouched so a
                # follow-up refinement keeps the original item (see set_user_anchor above)
                self.sessions.touch_results(
                    session_state,
                    [normalize_article_id(rid) for rid in result_ids],
                )
        log_payload = {
            "event": "chat",
            "request_id": request_id or "",
            "session_id": session_id or "",
            "query": query,
            "search_query": search_query,
            "search_query_raw": search_query_raw,
            "dense_query": dense_query,
            "sparse_query": sparse_query,
            "boosted_product_type": product_type,
            "boost_applied": boost_applied,
            "intent_hint": intent_hint,
            "analysis_debug": analysis_debug,
            "llm_filters_raw": llm_filters_raw,
            "llm_must_not_raw": llm_must_not_raw,
            "must_filters": must_filters,
            "soft_product_type": soft_product_type,
            "must_not_filters": must_not_filters,
            "dropped_filters": dropped_filters,
            "dropped_must_not": dropped_must_not,
            "filter_validation_debug": filter_validation_debug,
            "must_not_validation_debug": must_not_validation_debug,
            "retrieval_path": retrieval_path,
            "active_anchor_id": referential_anchor,
            "no_anchor": no_anchor,
            "graph_anchor_id": anchor_id,
            "graph_anchor_source": graph_anchor_source,
            "graph_neighbor_ids": graph_neighbor_ids,
            "graph_candidate_count": graph_candidate_count,
            "graph_filtered_count": graph_filtered_count,
            "graph_anchor_invalid": graph_anchor_invalid,
            "hybrid_result_count": hybrid_result_count,
            "has_image": bool(image),
            "result_ids": result_ids,
            "suggestions": suggestion_cards,
            "timing_ms": {
                "analysis": analysis_ms,
                "embedding": embed_ms,
            },
        }

        # ground the reply on only the top few even though more cards (items) are returned
        context = self._format_context(points[:self.gen_limit])
        if not context:
            return items, "", log_payload, False, None

        if suggestion_cards:
            # honest cross-category answer: the requested type does not match this request well, but
            # items in other categories do. The model infers the specifics from the query + the list.
            rtype = str(must_filters.get("product_type", ""))
            alt = "\n".join(f"- {c['name']} ({c['product_type']})" for c in suggestion_cards)
            system_prompt = (
                "You are a warm, concise fashion stylist talking to a customer. "
                f"Our {rtype} options do not really match what the customer asked for — the {rtype} cards "
                f"shown are only the closest {rtype} items. The ALTERNATIVES below, from OTHER categories, "
                "match the request much better.\n"
                f"In 1-2 warm sentences, gently acknowledge we don't have a great {rtype} for this specific "
                "request, then point the customer to the alternatives by name and category. Refer to pieces "
                "by name only; NEVER write article IDs, prices or measurements. Flowing prose, no lists.\n\n"
                f"{rtype.upper()} CARDS SHOWN:\n{context}\n\n"
                f"BETTER-MATCHING ALTERNATIVES (other categories):\n{alt}"
            )
        else:
            system_prompt = (
                "You are a warm, concise fashion stylist talking to a customer. "
                "The products below are already shown to the customer as clickable cards with full details, price and image — they ARE your recommendations.\n"
                "Write a SHORT, natural reply (1-2 sentences) that introduces the picks and why they work for the request or pair together. "
                "Refer to pieces by name only. NEVER write article IDs, prices, measurements or full spec lists — those live on the cards. "
                "Do not apologize and do not say nothing matches; the listed products are the answer. "
                "Write in flowing prose (no numbered or bulleted lists). Sound like a human stylist, not a catalogue.\n\n"
                f"PRODUCTS SHOWN:\n{context}"
            )
        full_prompt = f"{system_prompt}\n\nCustomer: {query}"
        return items, full_prompt, log_payload, True, None

    def stream_answer(self, prompt: str, image: Image.Image | None = None):
        return self.llm.generate_answer_stream(prompt, images=[image] if image else None)

    def prepare_chat(
        self,
        query: str,
        image: Image.Image | None,
        session_id: str | None,
        request_id: str | None,
        started_at: float,
        confirmed_intent: str | None = None,
        customer_id: str | None = None,
        selected_anchor_id: str | None = None,
        no_anchor: bool = False,
    ) -> tuple[List[dict], str, Dict[str, Any], bool, Dict[str, Any] | None]:
        return self._prepare_chat(query, image, session_id, request_id, started_at, confirmed_intent, customer_id, selected_anchor_id, no_anchor)

    def finalize_log(self, log_payload: Dict[str, Any], started_at: float, result_count: int) -> None:
        self._finalize_log(log_payload, started_at, result_count)

    async def chat(
        self,
        query: str,
        image: Image.Image | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        confirmed_intent: str | None = None,
        customer_id: str | None = None,
        selected_anchor_id: str | None = None,
        no_anchor: bool = False,
    ) -> tuple[str, List[dict], Dict[str, Any]]:
        started_at = time.perf_counter()
        items, full_prompt, log_payload, has_context, direct_response = self._prepare_chat(
            query=query,
            image=image,
            session_id=session_id,
            request_id=request_id,
            started_at=started_at,
            confirmed_intent=confirmed_intent,
            customer_id=customer_id,
            selected_anchor_id=selected_anchor_id,
            no_anchor=no_anchor,
        )
        active_anchor_id = log_payload.get("active_anchor_id", "")

        if direct_response is not None:
            intent_type = direct_response.get("type")
            if intent_type == INTENT_COMPOSITE:
                message = direct_response.get("message", COMPOSITE_INTENT_MESSAGE)
                self._finalize_log(log_payload, started_at, 0)
                return message, [], {
                    "intent": INTENT_COMPOSITE,
                    "intent_options": direct_response.get("intent_options", self.intent_confirm_options),
                    "intent_query": direct_response.get("intent_query", query),
                }
            if intent_type == INTENT_CHAT:
                message = self.llm.generate_chitchat_response(query)
                self._finalize_log(log_payload, started_at, 0)
                return message, [], {"intent": INTENT_CHAT}
            if intent_type == INTENT_NEEDS_REFERENCE:
                message = direct_response.get("message", NEEDS_REFERENCE_MESSAGE)
                self._finalize_log(log_payload, started_at, 0)
                return message, [], {"intent": INTENT_NEEDS_REFERENCE}

        if not has_context:
            self._finalize_log(log_payload, started_at, 0)
            intent = log_payload.get("intent_hint", "")
            message = NO_PAIRING_MESSAGE if intent == INTENT_GRAPH else NO_RESULTS_MESSAGE
            return message, [], {"intent": intent}

        message = self.llm.generate_answer(full_prompt, images=[image] if image else None)
        self._finalize_log(log_payload, started_at, len(items))
        extra = {"intent": log_payload.get("intent_hint", ""), "anchor_id": active_anchor_id}
        if log_payload.get("suggestions"):
            extra["suggestions"] = log_payload["suggestions"]
        return message, items, extra
