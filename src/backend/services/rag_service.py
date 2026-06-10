from __future__ import annotations

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
HARD_FILTER_KEYS = ("product_type", "colour_group")
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


class FashionRAGService:
    def __init__(self, embedder: QueryEncoder, store: QdrantStore, llm: QwenMultimodalService, catalog: FashionCatalog, limit: int = 5, personalization=None, sessions=None):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.catalog = catalog
        self.limit = max(1, int(limit))
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
        cascade: List[tuple[str, Dict[str, str]]] = []
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
        """Re-order a deep fused pool with a cross-encoder, fused via RRF so the bi-encoder
        retrieval rank still counts (measured: blend > pure rerank, md/exp_rerank.md). The
        cross-encoder reads (query, "<colour> <type>. <description>") jointly for relevance."""
        rr = self._get_reranker()
        if rr is None or not query or len(points) <= 1:
            return points[:top_k]
        docs: List[str] = []
        for p in points:
            pl = getattr(p, "payload", {}) or {}
            head = " ".join(x for x in [str(pl.get("colour_group", "")), str(pl.get("product_type", ""))] if x)
            docs.append(f"{head}. {pl.get('description', '')}".strip())
        scores = rr.score(query, docs)
        rerank_rank = {idx: r for r, idx in enumerate(sorted(range(len(points)), key=lambda i: scores[i], reverse=True))}
        blended = sorted(range(len(points)), key=lambda i: 1.0 / (60 + i) + 1.0 / (60 + rerank_rank[i]), reverse=True)
        return [points[i] for i in blended[:top_k]]

    def _image_knn_points(self, ref_emb, exclude_id: str, must_filters: Dict[str, str], must_not_filters: Dict[str, List[str]]) -> List[Any]:
        raw = self.store.hybrid_search(
            text_dense=None,
            image_dense=ref_emb,
            sparse_indices=None,
            sparse_values=None,
            limit=self.limit + 5,
            must_filters=must_filters or None,
            must_not_filters=must_not_filters,
        ) or []
        ex = normalize_article_id(exclude_id) if exclude_id else ""
        points = [
            p for p in raw
            if not ex or normalize_article_id(str((getattr(p, "payload", {}) or {}).get("article_id", ""))) != ex
        ]
        return points[: self.limit]

    def _filter_points(self, points: List[Any], must_filters: Dict[str, str], must_not_filters: Dict[str, List[str]]) -> List[Any]:
        filtered: List[Any] = []
        for point in points or []:
            payload = getattr(point, "payload", {}) or {}
            ok = True
            for key, value in (must_filters or {}).items():
                expected = normalize_text(str(value))
                actual = normalize_text(str(payload.get(key, "")))
                if expected and actual and expected not in actual and actual not in expected:
                    ok = False
                    break
            if not ok:
                continue
            excluded = False
            for key, values in (must_not_filters or {}).items():
                actual = normalize_text(str(payload.get(key, "")))
                for value in values or []:
                    expected = normalize_text(str(value))
                    if expected and actual and (expected in actual or actual in expected):
                        excluded = True
                        break
                if excluded:
                    break
            if excluded:
                continue
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
    ) -> tuple[List[dict], str, Dict[str, Any], bool, Dict[str, Any] | None]:
        analysis_started = time.perf_counter()
        analysis = self.llm.analyze_user_query(query, vocab=self.allowed_filters)
        analysis_ms = int((time.perf_counter() - analysis_started) * 1000)
        search_query = str(analysis.get("search_query_en", "") or "").strip() or query
        search_query_raw = search_query
        intent_hint = str(analysis.get("intent_hint", "") or "").strip()
        analysis_debug = analysis.get("debug", {}) if isinstance(analysis.get("debug"), dict) else {}
        intent_rules = analysis_debug.get("intent_rules", {}) if isinstance(analysis_debug, dict) else {}
        has_anchor_reference = bool(intent_rules.get("has_anchor_reference"))
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
        session_anchor = normalize_article_id(session_state.anchor_id) if session_state else ""
        classifier_intent = str(((analysis_debug or {}).get("intent_classifier") or {}).get("intent", "")).strip()
        requested_colour = str(must_filters.get("colour_group", "")).strip()
        referential_anchor = (
            normalize_article_id(extract_article_id_from_text(query) or "")
            or (normalize_article_id(selected_anchor_id) if (selected_anchor_id and has_anchor_reference) else "")
            or session_anchor
        )
        if classifier_intent == INTENT_VARIANT and requested_colour and (image is not None or referential_anchor):
            intent_hint = INTENT_VARIANT
            if isinstance(intent_rules, dict):
                intent_rules["rule_applied"] = "honor_color_variant_with_anchor"

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
        elif classifier_intent == INTENT_VARIANT and not (image is not None or referential_anchor):
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

        dense_query = search_query
        sparse_query = search_query
        product_type = str(must_filters.get("product_type", "")).strip()
        boost_applied = False
        if product_type:
            sparse_query = f"{product_type} {product_type} {product_type} {search_query}".strip()
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
        hybrid_result_count = 0
        retrieval_path = []

        if intent_hint == "graph_pairing":
            retrieval_path.append("intent_graph_pairing")
            anchor_id = extract_article_id_from_text(query)
            if anchor_id:
                graph_anchor_source = "query_reference"
            if not anchor_id and selected_anchor_id:
                anchor_id = normalize_article_id(selected_anchor_id)
                if anchor_id:
                    graph_anchor_source = "selected_anchor"
                    retrieval_path.append("anchor_from_selection")
            if not anchor_id:
                retrieval_path.append("anchor_from_hybrid_top1")
                anchor_points = self.store.hybrid_search(
                    text_dense=text_dense,
                    image_dense=image_dense,
                    sparse_indices=sparse_idx,
                    sparse_values=sparse_val,
                    limit=1,
                    must_filters=self._hard_filters(must_filters) or None,
                    must_not_filters=must_not_filters,
                )
                first = anchor_points[0] if anchor_points else None
                anchor_id = str((getattr(first, "payload", {}) or {}).get("article_id", "")) if first else ""
                if anchor_id:
                    graph_anchor_source = "hybrid_top1"

            anchor_id = normalize_article_id(anchor_id)
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
                    points = [p for p in points if get_slot(str((getattr(p, "payload", {}) or {}).get("product_type", ""))) == requested_slot]
                graph_filtered_count = len(points)

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
                elif selected_anchor_id and has_anchor_reference:
                    ref_id = normalize_article_id(selected_anchor_id)
                    ref_source = "selected_anchor"
                elif has_anchor_reference and session_anchor:
                    ref_id = session_anchor
                    ref_source = "session_anchor"
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
                ref_source = "query_reference" if extract_article_id_from_text(query) else ("selected_anchor" if (selected_anchor_id and has_anchor_reference) else "session_anchor")
                ref_emb = self.store.get_named_vector(ref_id, settings.vector_name_image)
                ref_type = self.catalog.get_meta(ref_id).get("product_type_name", "")
            if ref_emb and requested_colour:
                retrieval_path.append("color_variant_image_knn")
                anchor_id = ref_id
                graph_anchor_source = ref_source
                var_must = {"colour_group": requested_colour}
                if ref_type:
                    var_must["product_type"] = ref_type
                points = self._image_knn_points(ref_emb, ref_id, var_must, must_not_filters)
                hybrid_result_count = len(points)

        if not points and intent_hint == "graph_pairing":
            retrieval_path.append("no_graph_pairing")
        elif not points:
            do_rerank = settings.use_reranker and bool(query)
            pool = max(settings.rerank_candidate_depth, self.limit) if do_rerank else self.limit
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
                points = self._rerank_blend(query, points, self.limit)
                retrieval_path.append("cross_encoder_rerank")
            else:
                points = points[:self.limit]
            hybrid_result_count = len(points)

        items: List[dict] = []
        for point in points:
            payload = getattr(point, "payload", {}) or {}
            items.append({
                "article_id": str(payload.get("article_id", "")),
                "name": str(payload.get("prod_name", "") or payload.get("product_type", "") or ""),
                "product_type": str(payload.get("product_type", "")),
                "product_group": str(payload.get("product_group", "")),
                "colour_group": str(payload.get("colour_group", "")),
                "fit": str(payload.get("fit", "")),
                "occasion": str(payload.get("occasion", "")),
                "seasonality": str(payload.get("seasonality", "")),
                "description": str(payload.get("description", "")),
                "image_path": self._build_image_url(str(payload.get("article_id", ""))),
                "price": str(payload.get("price", "") or ""),
            })

        if self.personalization is not None and customer_id and self.personalization.has_profile(customer_id):
            order = self.personalization.rerank(customer_id, [it.get("article_id", "") for it in items])
            pos = {aid: i for i, aid in enumerate(order)}
            items.sort(key=lambda it: pos.get(it.get("article_id", ""), len(items)))
            retrieval_path.append("personalized_rerank")

        result_ids = [item.get("article_id", "") for item in items if item.get("article_id")]
        if session_state is not None and result_ids:
            self.sessions.touch_anchor(
                session_state,
                normalize_article_id(result_ids[0]),
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
            "graph_anchor_id": anchor_id,
            "graph_anchor_source": graph_anchor_source,
            "graph_neighbor_ids": graph_neighbor_ids,
            "graph_candidate_count": graph_candidate_count,
            "graph_filtered_count": graph_filtered_count,
            "hybrid_result_count": hybrid_result_count,
            "has_image": bool(image),
            "result_ids": result_ids,
            "timing_ms": {
                "analysis": analysis_ms,
                "embedding": embed_ms,
            },
        }

        context = self._format_context(points)
        if not context:
            return items, "", log_payload, False, None

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
    ) -> tuple[List[dict], str, Dict[str, Any], bool, Dict[str, Any] | None]:
        return self._prepare_chat(query, image, session_id, request_id, started_at, confirmed_intent, customer_id, selected_anchor_id)

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
        )

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
        return message, items, {"intent": log_payload.get("intent_hint", "")}
