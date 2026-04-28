from __future__ import annotations

import base64
import gc
import os
import time
import uuid
from datetime import datetime, timezone
from io import BytesIO
from typing import Dict, List

import torch
from PIL import Image

from src.backend.services.catalog import FashionCatalog
from src.backend.retrieval.embeddings import SiglipEmbeddingService
from src.backend.retrieval.llm import QwenMultimodalService
from src.backend.retrieval.qdrant import QdrantStore
from src.backend.models.schemas import ChatRequest
from src.backend.services.session_manager import SessionStore
from src.backend.core.config import Settings
from src.backend.core.utils import extract_article_id_from_text, normalize_article_id, normalize_text

INTENT_SIMILAR = "similar_items"
INTENT_GRAPH = "graph_pairing"
INTENT_VARIANT = "color_variant"

INTENT_UI_META = {
    INTENT_SIMILAR: {
        "label": "Similar Picks",
        "description": "Find the closest items by style, texture, and overall vibe from the current focus item.",
    },
    INTENT_GRAPH: {
        "label": "Outfit Pairing",
        "description": "Prioritize items with strong pairing likelihood based on the co-buy graph.",
    },
    INTENT_VARIANT: {
        "label": "Color Variants",
        "description": "Focus on nearby variants and alternate colors of the same design.",
    },
}


class FashionAssistantService:
    def __init__(
        self,
        settings: Settings,
        catalog: FashionCatalog,
        qdrant: QdrantStore,
        embedding: SiglipEmbeddingService,
        llm: QwenMultimodalService,
        sessions: SessionStore,
    ):
        self.settings = settings
        self.catalog = catalog
        self.qdrant = qdrant
        self.embedding = embedding
        self.llm = llm
        self.sessions = sessions

    @staticmethod
    def decode_image(base64_str: str | None) -> Image.Image | None:
        if not base64_str:
            return None
        try:
            payload = base64_str.split(",", 1)[1] if "," in base64_str else base64_str
            img = Image.open(BytesIO(base64.b64decode(payload))).convert("RGB")
            img.thumbnail((512, 512))
            return img
        except Exception:
            return None

    @staticmethod
    def _extract_article_id(point) -> str:
        payload = getattr(point, "payload", {}) or {}
        point_id = getattr(point, "id", "")
        return normalize_article_id(payload.get("article_id", point_id))

    def _pick_anchor_point(self, query_vector: List[float], query_filters: Dict[str, str] | None = None):
        points = self.qdrant.query(query_vector, limit=8, constraints=query_filters)
        if query_filters and not points:
            points = self.qdrant.query(query_vector, limit=8, constraints=None)

        for point in points:
            aid = self._extract_article_id(point)
            if aid:
                return point
        return points[0] if points else None

    def _build_item(self, payload: Dict | None, fallback_article_id: str = "") -> Dict:
        payload = payload or {}
        aid = normalize_article_id(payload.get("article_id", fallback_article_id))
        return self.catalog.build_item(aid, payload=payload)

    def _retrieve_items_by_ids(self, article_ids: List[str]) -> List[Dict]:
        if not article_ids:
            return []

        normalized = [normalize_article_id(a) for a in article_ids if normalize_article_id(a)]
        records = self.qdrant.retrieve_by_article_ids(normalized)

        payload_by_id: Dict[str, Dict] = {}
        for rec in records:
            payload = getattr(rec, "payload", {}) or {}
            rid = normalize_article_id(payload.get("article_id", getattr(rec, "id", "")))
            if rid:
                payload_by_id[rid] = payload

        items: List[Dict] = []
        for aid in normalized:
            payload = payload_by_id.get(aid)
            items.append(self._build_item(payload, fallback_article_id=aid))
        return items

    def _get_anchor_item_by_id(self, article_id: str) -> Dict | None:
        aid = normalize_article_id(article_id)
        if not aid:
            return None

        items = self._retrieve_items_by_ids([aid])
        if items and items[0].get("article_id"):
            return items[0]

        meta = self.catalog.get_meta(aid)
        if not meta:
            return None
        return self.catalog.build_item(aid)

    def _get_similar_items(
        self,
        query_vector: List[float],
        anchor_id: str,
        query_filters: Dict[str, str] | None,
        limit: int,
    ) -> List[Dict]:
        points = self.qdrant.query(query_vector, limit=limit + 12, constraints=query_filters)
        if query_filters and not points and not self.settings.strict_metadata_filters:
            points = self.qdrant.query(query_vector, limit=limit + 12, constraints=None)

        items: List[Dict] = []
        seen = {normalize_article_id(anchor_id)}
        for point in points:
            payload = getattr(point, "payload", {}) or {}
            aid = self._extract_article_id(point)
            if not aid or aid in seen:
                continue
            seen.add(aid)
            items.append(self._build_item(payload, fallback_article_id=aid))
            if len(items) >= limit:
                break
        return items

    @staticmethod
    def _matches_hard_filters(item: Dict, query_filters: Dict[str, str]) -> bool:
        if not query_filters:
            return True

        key_map = {
            "product_type": "product_type",
            "colour_group": "colour_group",
            "fit": "fit",
            "occasion": "occasion",
            "seasonality": "seasonality",
        }

        for filter_key, filter_value in query_filters.items():
            mapped = key_map.get(filter_key)
            if not mapped:
                continue
            expected = normalize_text(str(filter_value))
            actual = normalize_text(str(item.get(mapped, "")))
            if expected and expected != actual and expected not in actual and actual not in expected:
                return False
        return True

    @staticmethod
    def _matches_exclude_filters(item: Dict, exclude_filters: Dict[str, List[str]]) -> bool:
        if not exclude_filters:
            return True

        key_map = {
            "product_type": "product_type",
            "colour_group": "colour_group",
            "fit": "fit",
            "occasion": "occasion",
            "seasonality": "seasonality",
        }

        for filter_key, filter_values in exclude_filters.items():
            mapped = key_map.get(filter_key)
            if not mapped:
                continue
            if isinstance(filter_values, str):
                values = [filter_values]
            else:
                values = list(filter_values or [])

            actual = normalize_text(str(item.get(mapped, "")))
            for value in values:
                expected = normalize_text(str(value))
                if expected and actual and (expected == actual or expected in actual or actual in expected):
                    return False
        return True

    def _rank_items_by_target_slots(self, items: List[Dict], target_slots: List[str], limit: int) -> List[Dict]:
        if not items:
            return []

        selected: List[Dict] = []
        used_ids = set()

        by_slot: Dict[str, List[Dict]] = {}
        for item in items:
            slot = self.catalog.infer_item_slot(item)
            if slot:
                by_slot.setdefault(slot, []).append(item)

        for slot in target_slots:
            for item in by_slot.get(slot, []):
                aid = normalize_article_id(item.get("article_id", ""))
                if not aid or aid in used_ids:
                    continue
                selected.append(item)
                used_ids.add(aid)
                break
            if len(selected) >= limit:
                return selected[:limit]

        for item in items:
            aid = normalize_article_id(item.get("article_id", ""))
            if not aid or aid in used_ids:
                continue
            selected.append(item)
            used_ids.add(aid)
            if len(selected) >= limit:
                break
        return selected

    def _get_graph_items(self, anchor_id: str, user_query: str, limit: int) -> List[Dict]:
        target_slots = self.catalog.infer_target_slots(user_query, anchor_id=anchor_id)
        graph_ids = self.catalog.get_graph_multihop_outfit_ids(
            anchor_id,
            max_hops=self.settings.graph_max_hops,
            branch_per_hop=self.settings.graph_branch_per_hop,
            preferred_min_weight=self.settings.graph_preferred_min_weight,
            hard_min_weight=self.settings.graph_hard_min_weight,
            limit=max(limit * 3, 9),
            target_slots=target_slots,
        )
        items = self._retrieve_items_by_ids(graph_ids)
        items = self._rank_items_by_target_slots(items, target_slots=target_slots, limit=max(limit * 2, 8))
        items = self.catalog.filter_items_by_target(items, user_query)
        return items[:limit]

    def _get_graph_items_via_proxy(self, anchor_item: Dict, user_query: str, limit: int) -> List[Dict]:
        anchor_id = normalize_article_id(anchor_item.get("article_id", ""))
        anchor_hint = f"{anchor_item.get('product_type', '')} {anchor_item.get('colour_group', '')} {anchor_item.get('description', '')}".strip()
        if not anchor_hint:
            return []

        proxy_vector = self.embedding.encode(anchor_hint, image=None)
        proxy_points = self.qdrant.query(proxy_vector, limit=12, constraints=None)

        seen = {anchor_id}
        collected: List[Dict] = []
        for point in proxy_points:
            proxy_id = self._extract_article_id(point)
            if not proxy_id or proxy_id in seen:
                continue

            proxy_items = self._get_graph_items(proxy_id, user_query, limit=max(limit, 4))
            for item in proxy_items:
                iid = normalize_article_id(item.get("article_id", ""))
                if not iid or iid in seen:
                    continue
                seen.add(iid)
                collected.append(item)
                if len(collected) >= limit:
                    return collected

        return collected

    def _route_intent(self, user_query: str, has_image: bool, anchor_item: Dict | None = None) -> tuple[str, bool]:
        query = normalize_text(user_query)

        if not query:
            return INTENT_SIMILAR if has_image else INTENT_SIMILAR, False

        if not self.settings.use_llm_router:
            return INTENT_SIMILAR if has_image else INTENT_SIMILAR, False

        llm_intent = self.llm.route_intent(user_query, anchor_item=anchor_item)
        if llm_intent in {INTENT_SIMILAR, INTENT_GRAPH, INTENT_VARIANT}:
            if llm_intent == INTENT_SIMILAR and anchor_item is not None:
                pairing_flag = self.llm.detect_pairing_request(user_query, anchor_item=anchor_item)
                if pairing_flag is True:
                    return INTENT_GRAPH, True
            return llm_intent, False
        return INTENT_SIMILAR if has_image else INTENT_SIMILAR, False

    def _resolve_query_understanding(
        self,
        raw_query: str,
        rewritten_query: str | None,
        anchor_item: Dict | None = None,
    ) -> Dict[str, Dict[str, str] | Dict[str, List[str]] | str]:
        filters_raw = self.catalog.parse_query_filters(raw_query)
        filters_rewritten = self.catalog.parse_query_filters(rewritten_query or "") if rewritten_query else {}
        llm_constraints = self.llm.extract_query_constraints(raw_query, anchor_item=anchor_item) or {}

        filters_llm_all = llm_constraints.get("include_filters") if isinstance(llm_constraints.get("include_filters"), dict) else {}
        filters_llm = {
            k: str(v).strip()
            for k, v in filters_llm_all.items()
            if str(v).strip()
        }
        exclude_filters = llm_constraints.get("exclude_filters") if isinstance(llm_constraints.get("exclude_filters"), dict) else {}

        merged = dict(filters_raw)
        for key, value in filters_rewritten.items():
            merged[key] = value
        for key, value in filters_llm.items():
            merged[key] = value

        search_query = rewritten_query or str(llm_constraints.get("search_query", "") or "").strip() or raw_query
        intent_hint = str(llm_constraints.get("intent_hint", "") or "").strip()

        return {
            "search_query": search_query,
            "filters_raw": filters_raw,
            "filters_rewritten": filters_rewritten,
            "filters_llm": filters_llm,
            "filters_final": merged,
            "exclude_filters": exclude_filters,
            "intent_hint": intent_hint,
            "llm_notes": str(llm_constraints.get("notes", "") or "").strip(),
            "llm_model": str(llm_constraints.get("model", "") or "").strip(),
        }

    @staticmethod
    def _normalize_joint_weights(text_weight: float, image_weight: float) -> tuple[float, float]:
        tw = max(0.0, float(text_weight))
        iw = max(0.0, float(image_weight))
        total = tw + iw
        if total <= 0:
            return 0.5, 0.5
        return tw / total, iw / total

    def _resolve_joint_weights(
        self,
        req: ChatRequest,
        user_query: str,
        query_filters: Dict[str, str],
        intent: str,
        has_image: bool,
    ) -> tuple[float, float, str]:
        if not has_image:
            return 1.0, 0.0, "text_only"

        if req.embedding_text_weight is not None or req.embedding_image_weight is not None:
            tw, iw = self._normalize_joint_weights(
                req.embedding_text_weight if req.embedding_text_weight is not None else 0.5,
                req.embedding_image_weight if req.embedding_image_weight is not None else 0.5,
            )
            return tw, iw, "request_override"

        query = normalize_text(user_query)
        if not query:
            return 0.1, 0.9, "auto_image_dominant_no_text"

        if intent == INTENT_VARIANT:
            return 0.7, 0.3, "auto_variant_text_priority"

        if query_filters:
            return 0.6, 0.4, "auto_metadata_filter_priority"

        visual_cues = ["nhu anh", "giong anh", "same look", "look nay", "style nay", "na na"]
        if any(k in query for k in visual_cues):
            return 0.25, 0.75, "auto_visual_priority"

        if intent == INTENT_GRAPH:
            return 0.45, 0.55, "auto_graph_balanced"

        return 0.4, 0.6, "auto_default_visual_bias"

    def _build_similar_vector(
        self,
        user_query: str,
        image: Image.Image | None,
        anchor_item: Dict,
        text_weight: float | None = None,
        image_weight: float | None = None,
    ) -> List[float]:
        if image is not None:
            return self.embedding.encode(
                user_query or "",
                image=image,
                text_weight=text_weight,
                image_weight=image_weight,
            )

        anchor_hint = f"{anchor_item.get('product_type', '')} {anchor_item.get('colour_group', '')} {anchor_item.get('description', '')}".strip()
        merged_text = f"{user_query or ''} {anchor_hint}".strip() or anchor_hint or "fashion item"
        return self.embedding.encode(merged_text, image=None)

    def _build_summary_images(self, items: List[Dict]) -> List[str]:
        paths = []
        for item in items:
            article_id = normalize_article_id(item.get("article_id", ""))
            if not article_id:
                continue
            p = self.catalog.article_image_path(article_id)
            if os.path.exists(p):
                paths.append(p)
            if len(paths) >= self.settings.max_vision_images:
                break
        return paths

    @staticmethod
    def _resolve_ui_item_limit(req: ChatRequest) -> int:
        if req.max_ui_items is None:
            return 10
        try:
            value = int(req.max_ui_items)
        except (TypeError, ValueError):
            return 10
        return max(1, min(value, 20))

    @staticmethod
    def _image_url_for_article(article_id: str) -> str:
        aid = normalize_article_id(article_id)
        if not aid:
            return ""
        return f"/images/{aid[:3]}/{aid}.jpg"

    @staticmethod
    def _intent_info(intent: str) -> Dict[str, str]:
        return dict(
            INTENT_UI_META.get(
                intent,
                {
                    "label": "Recommendations",
                    "description": "Curated retrieved items to help you refine the next step.",
                },
            )
        )

    def _build_anchor_options(self, items: List[Dict], limit: int) -> List[Dict]:
        options: List[Dict] = []
        for item in items[:limit]:
            aid = normalize_article_id(item.get("article_id", ""))
            if not aid:
                continue
            options.append(
                {
                    "article_id": aid,
                    "product_type": item.get("product_type", ""),
                    "colour_group": item.get("colour_group", ""),
                }
            )
        return options

    def _build_ui_cards(self, items: List[Dict], anchor_id: str, limit: int) -> List[Dict]:
        cards: List[Dict] = []
        anchor = normalize_article_id(anchor_id)
        for rank, item in enumerate(items[:limit], 1):
            aid = normalize_article_id(item.get("article_id", ""))
            if not aid:
                continue
            cards.append(
                {
                    "rank": rank,
                    "article_id": aid,
                    "title": item.get("product_type") or "Item",
                    "subtitle": item.get("colour_group") or "",
                    "fit": item.get("fit") or "",
                    "occasion": item.get("occasion") or "",
                    "seasonality": item.get("seasonality") or "",
                    "description": item.get("description") or "",
                    "image_url": self._image_url_for_article(aid),
                    "is_anchor": aid == anchor,
                }
            )
        return cards

    def _build_quick_actions(self, intent: str, anchor_id: str, items: List[Dict]) -> List[str]:
        anchor = normalize_article_id(anchor_id)
        anchor_ref = f"#{anchor}" if anchor else "this item"

        if intent == INTENT_GRAPH:
            actions = [
                f"Find shoes that match {anchor_ref}",
                f"Find pants that pair with {anchor_ref}",
                f"Find outerwear that complements {anchor_ref}",
            ]
        elif intent == INTENT_VARIANT:
            actions = [
                f"Show darker color options for {anchor_ref}",
                f"Show lighter color options for {anchor_ref}",
                f"Show winter-ready variants of {anchor_ref}",
            ]
        else:
            actions = [
                f"Build an outfit around {anchor_ref}",
                f"Find even closer matches to {anchor_ref}",
                f"Find a minimal version of {anchor_ref}",
            ]

        for item in items:
            aid = normalize_article_id(item.get("article_id", ""))
            if aid and aid != anchor:
                actions.append(f"Compare with #{aid}")
                break

        deduped: List[str] = []
        for action in actions:
            if action not in deduped:
                deduped.append(action)
        return deduped[:6]

    def _compose_response(
        self,
        req: ChatRequest,
        answer: str,
        items: List[Dict],
        intent: str,
        anchor_article_id: str,
        session_id: str,
        started_at: float,
        request_id: str,
        trace: Dict | None = None,
    ) -> Dict:
        response_mode = (req.response_mode or "rich").strip().lower()
        if response_mode not in {"rich", "compact"}:
            response_mode = "rich"

        safe_items = items or []
        ui_limit = self._resolve_ui_item_limit(req)
        intent_info = self._intent_info(intent)

        payload = {
            "answer": answer,
            "items": safe_items,
            "intent": intent,
            "intent_info": intent_info,
            "anchor_article_id": normalize_article_id(anchor_article_id),
            "session_id": session_id,
            "meta": {
                "request_id": request_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "latency_ms": int((time.perf_counter() - started_at) * 1000),
                "result_count": len(safe_items),
                "response_mode": response_mode,
            },
        }

        if response_mode == "rich":
            payload["ui"] = {
                "cards": self._build_ui_cards(safe_items, anchor_article_id, ui_limit),
                "anchor_options": self._build_anchor_options(safe_items, ui_limit),
                "quick_actions": self._build_quick_actions(intent, anchor_article_id, safe_items),
                "intent_chip": intent_info.get("label", ""),
            }

        if req.include_debug and trace is not None:
            payload["debug"] = trace

        return payload

    def frontend_bootstrap(self) -> Dict:
        return {
            "app": {
                "name": "Atelier mRAG",
                "tagline": "Retro Vintage Multimodal Styling Console",
                "api_version": "2",
                "theme": "retro-vintage",
            },
            "tech_stack": [
                {"name": "React 18", "role": "Component UI runtime"},
                {"name": "Next.js App Router", "role": "Route-first architecture and image optimization"},
                {"name": "Tailwind CSS", "role": "Utility-first styling"},
                {"name": "shadcn/ui", "role": "Headless UI primitives with full code ownership"},
                {"name": "Zustand", "role": "Lightweight global state for chat and anchor context"},
                {"name": "Framer Motion", "role": "Interaction animation and transition UX"},
                {"name": "React Query", "role": "API cache, request states, and mutation orchestration"},
                {"name": "FastAPI", "role": "Backend orchestration API"},
                {"name": "Qdrant", "role": "Vector retrieval store"},
                {"name": "Qwen2-VL", "role": "Multimodal reasoning and summary"},
                {"name": "SigLIP", "role": "Joint image-text embedding"},
                {"name": "Graph RAG", "role": "Outfit pairing via co-buy graph"},
            ],
            "capabilities": {
                "image_upload": True,
                "anchor_selection": True,
                "weighted_joint_embedding": True,
                "graph_pairing": True,
                "multi_hop_graph": True,
                "query_understanding_filters": True,
                "query_rewrite": self.settings.use_query_rewrite,
                "quick_actions": True,
            },
            "defaults": {
                "topk_similar": self.settings.topk_similar,
                "topk_graph": self.settings.topk_graph,
                "topk_variants": self.settings.topk_variants,
                "max_vision_images": self.settings.max_vision_images,
                "max_ui_items": 10,
            },
            "intents": [
                {"id": intent_id, **meta}
                for intent_id, meta in INTENT_UI_META.items()
            ],
            "suggested_prompts": [
                "Find pants that match this top",
                "Show another color of #article_id",
                "Build a smart-casual outfit from this item",
                "Find a cleaner minimal version of this look",
            ],
        }

    def handle_chat(self, req: ChatRequest) -> Dict:
        started_at = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]

        if self.settings.clean_cuda_cache_each_request:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        user_query = (req.text or "").strip()
        image = self.decode_image(req.image)

        session_id, session = self.sessions.get_or_create(req.session_id)
        if req.new_image_context and image is not None:
            self.sessions.reset(session)

        selected_anchor_id = normalize_article_id(req.selected_anchor_id)
        text_anchor_id = extract_article_id_from_text(user_query)
        explicit_anchor_id = selected_anchor_id or text_anchor_id

        if not user_query and image is None and not explicit_anchor_id and not session.anchor_id:
            return self._compose_response(
                req=req,
                answer="Please type a request or upload an image so I can suggest matching fashion items.",
                items=[],
                intent=INTENT_SIMILAR,
                anchor_article_id="",
                session_id=session_id,
                started_at=started_at,
                request_id=request_id,
            )

        initial_query_filters = self.catalog.parse_query_filters(user_query)
        trace = {
            "query_understanding": {
                "search_query": user_query,
                "filters_raw": initial_query_filters,
                "filters_rewritten": {},
                "filters_llm": {},
                "filters_final": dict(initial_query_filters),
                "exclude_filters": {},
                "intent_hint": "",
            },
            "anchor_source": "",
            "intent_router": "llm_only",
            "intent_initial": "",
            "intent_final": "",
            "intent_override_reason": "",
            "intent_refine_pairing": False,
            "graph_used_proxy": False,
            "query_rewrite_applied": False,
            "joint_weights": {},
        }

        anchor_item: Dict | None = None
        anchor_id = ""

        if image is not None:
            image_anchor_vector = self.embedding.encode("", image=image, text_weight=0.0, image_weight=1.0)
            anchor_point = self._pick_anchor_point(image_anchor_vector, query_filters=None)
            if anchor_point is None:
                return self._compose_response(
                    req=req,
                    answer="I could not find a valid focus product from the uploaded image.",
                    items=[],
                    intent=INTENT_SIMILAR,
                    anchor_article_id="",
                    session_id=session_id,
                    started_at=started_at,
                    request_id=request_id,
                    trace=trace,
                )
            payload = getattr(anchor_point, "payload", {}) or {}
            anchor_id = self._extract_article_id(anchor_point)
            anchor_item = self._build_item(payload, fallback_article_id=anchor_id)
            trace["anchor_source"] = "image"
        else:
            if explicit_anchor_id:
                anchor_item = self._get_anchor_item_by_id(explicit_anchor_id)
                if anchor_item is None:
                    return self._compose_response(
                        req=req,
                        answer=f"I could not find product #{explicit_anchor_id}. Please choose another item or upload a new image.",
                        items=[],
                        intent=INTENT_SIMILAR,
                        anchor_article_id="",
                        session_id=session_id,
                        started_at=started_at,
                        request_id=request_id,
                        trace=trace,
                    )
                trace["anchor_source"] = "explicit"

            if anchor_item is None and session.anchor_id:
                anchor_item = self._get_anchor_item_by_id(session.anchor_id)
                if anchor_item is not None:
                    trace["anchor_source"] = "session"

            if anchor_item is None and user_query:
                text_vector = self.embedding.encode(user_query, image=None)
                anchor_point = self._pick_anchor_point(text_vector, query_filters=initial_query_filters)
                if anchor_point is not None:
                    payload = getattr(anchor_point, "payload", {}) or {}
                    anchor_id = self._extract_article_id(anchor_point)
                    anchor_item = self._build_item(payload, fallback_article_id=anchor_id)
                    trace["anchor_source"] = "text_retrieval"

            if anchor_item is None:
                return self._compose_response(
                    req=req,
                    answer="I do not have a focus item yet. Upload an image or choose one from the item list to continue.",
                    items=[],
                    intent=INTENT_SIMILAR,
                    anchor_article_id="",
                    session_id=session_id,
                    started_at=started_at,
                    request_id=request_id,
                    trace=trace,
                )

            anchor_id = normalize_article_id(anchor_item.get("article_id", ""))

        if not anchor_item or not anchor_id:
            return self._compose_response(
                req=req,
                answer="I could not resolve a valid focus item for this request.",
                items=[],
                intent=INTENT_SIMILAR,
                anchor_article_id="",
                session_id=session_id,
                started_at=started_at,
                request_id=request_id,
                trace=trace,
            )

        intent, intent_refine_pairing = self._route_intent(user_query, has_image=image is not None, anchor_item=anchor_item)
        trace["intent_refine_pairing"] = intent_refine_pairing
        if intent_refine_pairing:
            trace["intent_override_reason"] = "llm_pairing_refine"
        trace["intent_initial"] = intent

        rewritten_query = None
        if self.settings.use_query_rewrite and image is not None and normalize_text(user_query):
            rewritten_query = self.llm.rewrite_search_query(
                user_query=user_query,
                anchor_item=anchor_item,
                intent_label=intent,
                image=image,
            )
            if rewritten_query:
                trace["query_rewrite_applied"] = True

        query_understanding = self._resolve_query_understanding(user_query, rewritten_query, anchor_item=anchor_item)
        search_query = str(query_understanding.get("search_query", "") or "")
        query_filters = dict(query_understanding.get("filters_final", {}) or {})
        exclude_filters = dict(query_understanding.get("exclude_filters", {}) or {})

        intent_hint = str(query_understanding.get("intent_hint", "") or "")
        if intent_hint in {INTENT_SIMILAR, INTENT_GRAPH, INTENT_VARIANT} and intent_hint != intent:
            intent = intent_hint
            trace["intent_override_reason"] = "llm_constraints_intent_hint"

        trace["query_understanding"] = query_understanding

        text_weight, image_weight, weight_source = self._resolve_joint_weights(
            req,
            user_query=search_query,
            query_filters=query_filters,
            intent=intent,
            has_image=image is not None,
        )
        trace["joint_weights"] = {
            "text": round(text_weight, 4),
            "image": round(image_weight, 4),
            "source": weight_source,
        }
        routed_items: List[Dict] = []

        if intent == INTENT_GRAPH:
            routed_items = self._get_graph_items(anchor_id, search_query, self.settings.topk_graph)
            if query_filters or exclude_filters:
                routed_items = [
                    it
                    for it in routed_items
                    if self._matches_hard_filters(it, query_filters)
                    and self._matches_exclude_filters(it, exclude_filters)
                ]
            if not routed_items:
                routed_items = self._get_graph_items_via_proxy(anchor_item, search_query, self.settings.topk_graph)
                if query_filters or exclude_filters:
                    routed_items = [
                        it
                        for it in routed_items
                        if self._matches_hard_filters(it, query_filters)
                        and self._matches_exclude_filters(it, exclude_filters)
                    ]
                trace["graph_used_proxy"] = bool(routed_items)
            if not routed_items:
                intent = INTENT_SIMILAR

        elif intent == INTENT_VARIANT:
            variant_ids = self.catalog.get_color_variant_ids(anchor_id, self.settings.topk_variants)
            routed_items = self._retrieve_items_by_ids(variant_ids)
            if query_filters or exclude_filters:
                routed_items = [
                    it
                    for it in routed_items
                    if self._matches_hard_filters(it, query_filters)
                    and self._matches_exclude_filters(it, exclude_filters)
                ]
            if not routed_items:
                intent = INTENT_SIMILAR

        if intent == INTENT_SIMILAR:
            sim_vector = self._build_similar_vector(
                search_query,
                image,
                anchor_item,
                text_weight=text_weight,
                image_weight=image_weight,
            )
            routed_items = self._get_similar_items(
                sim_vector,
                anchor_id=anchor_id,
                query_filters=query_filters,
                limit=self.settings.topk_similar,
            )
            if query_filters or exclude_filters:
                routed_items = [
                    it
                    for it in routed_items
                    if self._matches_hard_filters(it, query_filters)
                    and self._matches_exclude_filters(it, exclude_filters)
                ]

        deduped: List[Dict] = []
        seen_ids = set()
        for item in [anchor_item] + routed_items:
            aid = normalize_article_id(item.get("article_id", ""))
            if not aid or aid in seen_ids:
                continue
            is_anchor = aid == anchor_id
            if exclude_filters and not self._matches_exclude_filters(item, exclude_filters):
                continue
            if (not is_anchor) and query_filters and not self._matches_hard_filters(item, query_filters):
                continue
            seen_ids.add(aid)
            deduped.append(item)

        summary_images = self._build_summary_images(deduped)
        answer = self.llm.summarize(
            user_query=user_query or search_query,
            intent_label=intent,
            anchor_item=anchor_item,
            result_items=deduped,
            image_paths=summary_images,
            extra_images=[image] if image is not None else None,
        )
        trace["intent_final"] = intent
        trace["result_count"] = len(deduped)

        self.sessions.touch_anchor(session, anchor_id=anchor_id, item_ids=[it["article_id"] for it in deduped])

        return self._compose_response(
            req=req,
            answer=answer,
            items=deduped,
            intent=intent,
            anchor_article_id=anchor_id,
            session_id=session_id,
            started_at=started_at,
            request_id=request_id,
            trace=trace,
        )
