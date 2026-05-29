from __future__ import annotations

import time
from typing import Any, Dict, List

from PIL import Image

from src.backend.core.config import settings
from src.backend.core.query_logger import append_log
from src.backend.core.utils import extract_article_id_from_text, normalize_article_id, normalize_text
from src.backend.retrieval.embeddings import HybridEmbeddingService
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

INTENT_CONFIRM_OPTIONS = [INTENT_SIMILAR, INTENT_GRAPH, INTENT_VARIANT]
COMPOSITE_INTENT_MESSAGE = (
    "Sorry, I'm a little confused by your last request. "
    "Could you help me confirm your intent so I can assist you better?"
)


class FashionRAGService:
    def __init__(self, embedder: HybridEmbeddingService, store: QdrantStore, llm: QwenMultimodalService, catalog: FashionCatalog, limit: int = 5):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.catalog = catalog
        self.limit = max(1, int(limit))
        self.intent_confirm_options = list(INTENT_CONFIRM_OPTIONS)
        self.allowed_filters = self._build_allowed_filters()

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
        value_lower = value_str.lower()
        matched = next((v for v in allowed if v.lower() == value_lower), "")
        if not matched:
            if dropped is not None:
                dropped.append(f"{key}={value_str}")
            print(f"[WARNING] Dropping invalid filter: {key}={value_str}")
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
        for idx, point in enumerate(points or [], 1):
            item = getattr(point, "payload", {}) or {}
            article_id = str(item.get("article_id", ""))
            name = str(item.get("prod_name", "") or item.get("product_type", "") or "")
            category = f"{item.get('product_type', '')} - {item.get('product_group', '')}".strip(" -")
            description = str(item.get("description", ""))
            colour = str(item.get("colour_group", ""))
            fit = str(item.get("fit", ""))
            occasion = str(item.get("occasion", ""))
            part = (
                f"[{idx}] Article ID: {article_id}\n"
                f"Name: {name}\n"
                f"Category: {category}\n"
                f"Description: {description}\n"
                f"Attributes: Color: {colour}, Fit: {fit}, Occasion: {occasion}\n"
            )
            context_parts.append(part)
        return "\n---\n".join(context_parts)

    def _build_image_url(self, article_id: str) -> str:
        aid = normalize_article_id(article_id)
        if not aid:
            return ""
        return f"/images/{aid[:3]}/{aid}.jpg"

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
    ) -> tuple[List[dict], str, Dict[str, Any], bool, Dict[str, Any] | None]:
        analysis_started = time.perf_counter()
        analysis = self.llm.analyze_user_query(query)
        analysis_ms = int((time.perf_counter() - analysis_started) * 1000)
        search_query = str(analysis.get("search_query_en", "") or "").strip() or query
        search_query_raw = search_query
        intent_hint = str(analysis.get("intent_hint", "") or "").strip()
        analysis_debug = analysis.get("debug", {}) if isinstance(analysis.get("debug"), dict) else {}
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

        confirmed_value = str(confirmed_intent or "").strip().lower()
        if confirmed_value in self.intent_confirm_options:
            intent_hint = confirmed_value
            if isinstance(analysis_debug, dict):
                analysis_debug["intent_override"] = {
                    "source": "user_confirmed",
                    "value": confirmed_value,
                }

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
        dense_vec, sparse_idx, sparse_val = self.embedder.encode_hybrid(dense_query, sparse_query, image=image)
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
            if not anchor_id:
                retrieval_path.append("anchor_from_hybrid_top1")
                anchor_points = self.store.hybrid_search(
                    dense_vector=dense_vec,
                    sparse_indices=sparse_idx,
                    sparse_values=sparse_val,
                    limit=1,
                    must_filters=must_filters,
                    must_not_filters=must_not_filters,
                )
                first = anchor_points[0] if anchor_points else None
                anchor_id = str((getattr(first, "payload", {}) or {}).get("article_id", "")) if first else ""
                if anchor_id:
                    graph_anchor_source = "hybrid_top1"

            anchor_id = normalize_article_id(anchor_id)
            if anchor_id:
                retrieval_path.append("graph_neighbors")
                neighbor_ids = self.catalog.get_graph_diverse_neighbors(
                    anchor_id,
                    limit=self.limit,
                    max_per_pt=settings.graph_max_per_pt,
                    preferred_min_weight=settings.graph_preferred_min_weight,
                    hard_min_weight=settings.graph_hard_min_weight,
                )
                graph_neighbor_ids = list(neighbor_ids or [])
                graph_candidate_count = len(graph_neighbor_ids)
                points = self.store.retrieve_by_article_ids(graph_neighbor_ids)
                points = self._filter_points(points, must_filters, must_not_filters)
                graph_filtered_count = len(points)

        if not points:
            retrieval_path.append("fallback_hybrid_search" if intent_hint == "graph_pairing" else "hybrid_search")
            results = self.store.hybrid_search(
                dense_vector=dense_vec,
                sparse_indices=sparse_idx,
                sparse_values=sparse_val,
                limit=self.limit,
                must_filters=must_filters,
                must_not_filters=must_not_filters,
            )
            points = results or []
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

        result_ids = [item.get("article_id", "") for item in items if item.get("article_id")]
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
            return items, "", log_payload, False

        system_prompt = (
            "You are a premium AI fashion stylist. The context below lists real products from inventory. "
            "Rules:\n"
            "1. Recommend only products that appear in the context.\n"
            "2. Always include the Article ID for each product you mention.\n"
            "3. If nothing matches the request, say so clearly.\n"
            "4. Keep the response concise, professional, and style-focused.\n\n"
            f"CONTEXT:\n{context}"
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
    ) -> tuple[List[dict], str, Dict[str, Any], bool, Dict[str, Any] | None]:
        return self._prepare_chat(query, image, session_id, request_id, started_at, confirmed_intent)

    def finalize_log(self, log_payload: Dict[str, Any], started_at: float, result_count: int) -> None:
        self._finalize_log(log_payload, started_at, result_count)

    async def chat(
        self,
        query: str,
        image: Image.Image | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
        confirmed_intent: str | None = None,
    ) -> tuple[str, List[dict], Dict[str, Any]]:
        started_at = time.perf_counter()
        items, full_prompt, log_payload, has_context, direct_response = self._prepare_chat(
            query=query,
            image=image,
            session_id=session_id,
            request_id=request_id,
            started_at=started_at,
            confirmed_intent=confirmed_intent,
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

        if not has_context:
            self._finalize_log(log_payload, started_at, 0)
            return "I could not find any matching items in the current inventory.", [], {
                "intent": log_payload.get("intent_hint", ""),
            }

        message = self.llm.generate_answer(full_prompt, images=[image] if image else None)
        self._finalize_log(log_payload, started_at, len(items))
        return message, items, {"intent": log_payload.get("intent_hint", "")}
