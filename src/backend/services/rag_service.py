from __future__ import annotations

import time
from typing import Any, Dict, List

from PIL import Image

from src.backend.core.config import settings
from src.backend.core.query_logger import append_log
from src.backend.core.utils import extract_article_id_from_text, normalize_article_id, normalize_text
from src.backend.retrieval.embeddings import HybridEmbeddingService
from src.backend.retrieval.llm import QwenMultimodalService
from src.backend.retrieval.qdrant import QdrantStore
from src.backend.services.catalog import FashionCatalog


class FashionRAGService:
    def __init__(self, embedder: HybridEmbeddingService, store: QdrantStore, llm: QwenMultimodalService, catalog: FashionCatalog, limit: int = 5):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.catalog = catalog
        self.limit = max(1, int(limit))
        self.filter_maps = self._build_filter_maps()

    def _build_filter_maps(self) -> Dict[str, Dict[str, str]]:
        maps: Dict[str, Dict[str, str]] = {}

        def build_map(values: List[tuple[str, str]]) -> Dict[str, str]:
            out: Dict[str, str] = {}
            for normalized, original in values:
                if not normalized or not original:
                    continue
                out[normalized] = original
            return out

        maps["product_type"] = build_map(getattr(self.catalog, "product_type_values", []))
        maps["colour_group"] = build_map(getattr(self.catalog, "color_values", []))
        maps["fit"] = build_map(getattr(self.catalog, "fit_values", []))
        maps["occasion"] = build_map(getattr(self.catalog, "occasion_values", []))

        seasonality_map: Dict[str, str] = {}
        meta = getattr(self.catalog, "df_meta", None)
        if meta is not None and "seasonality" in meta.columns:
            for value in meta["seasonality"].dropna().unique():
                raw = str(value).strip()
                if not raw:
                    continue
                seasonality_map[normalize_text(raw)] = raw
        maps["seasonality"] = seasonality_map
        return maps

    def _validate_filter_value(self, key: str, value: str) -> str:
        if not value:
            return ""
        normalized = normalize_text(value)
        if not normalized:
            return ""
        mapping = self.filter_maps.get(key, {})
        return mapping.get(normalized, "")

    def _validate_filters(self, filters: Dict[str, str]) -> Dict[str, str]:
        validated: Dict[str, str] = {}
        for key, value in (filters or {}).items():
            cleaned = self._validate_filter_value(key, str(value).strip())
            if cleaned:
                validated[key] = cleaned
        return validated

    def _validate_must_not(self, filters: Dict[str, List[str] | str]) -> Dict[str, List[str]]:
        validated: Dict[str, List[str]] = {}
        for key, value in (filters or {}).items():
            if isinstance(value, str):
                values = [value]
            else:
                values = list(value or [])
            cleaned_values: List[str] = []
            for item in values:
                cleaned = self._validate_filter_value(key, str(item).strip())
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

    async def chat(
        self,
        query: str,
        image: Image.Image | None = None,
        session_id: str | None = None,
        request_id: str | None = None,
    ) -> tuple[str, List[dict]]:
        started_at = time.perf_counter()
        analysis = self.llm.analyze_user_query(query)
        search_query = str(analysis.get("search_query_en", "") or "").strip() or query
        intent_hint = str(analysis.get("intent_hint", "") or "").strip()
        must_filters = self._validate_filters(analysis.get("must_filters", {}))
        must_not_filters = self._validate_must_not(analysis.get("must_not_filters", {}))
        dense_vec, sparse_idx, sparse_val = self.embedder.encode_hybrid(search_query, search_query, image=image)
        points: List[Any] = []
        anchor_id = ""

        if intent_hint == "graph_pairing":
            anchor_id = extract_article_id_from_text(query)
            if not anchor_id:
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

            anchor_id = normalize_article_id(anchor_id)
            if anchor_id:
                neighbor_ids = self.catalog.get_graph_neighbor_ids(
                    anchor_id,
                    limit=self.limit,
                    preferred_min_weight=settings.graph_preferred_min_weight,
                    hard_min_weight=settings.graph_hard_min_weight,
                )
                points = self.store.retrieve_by_article_ids(neighbor_ids)
                points = self._filter_points(points, must_filters, must_not_filters)

        if not points:
            results = self.store.hybrid_search(
                dense_vector=dense_vec,
                sparse_indices=sparse_idx,
                sparse_values=sparse_val,
                limit=self.limit,
                must_filters=must_filters,
                must_not_filters=must_not_filters,
            )
            points = results or []
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

        context = self._format_context(points)
        if not context:
            append_log(
                settings.log_dir,
                {
                    "event": "chat",
                    "request_id": request_id or "",
                    "session_id": session_id or "",
                    "query": query,
                    "search_query": search_query,
                    "intent_hint": intent_hint,
                    "must_filters": must_filters,
                    "must_not_filters": must_not_filters,
                    "result_count": 0,
                    "graph_anchor_id": anchor_id,
                    "graph_used": intent_hint == "graph_pairing" and bool(anchor_id),
                    "has_image": bool(image),
                    "latency_ms": int((time.perf_counter() - started_at) * 1000),
                },
            )
            return "I could not find any matching items in the current inventory.", []

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
        message = self.llm.generate_answer(full_prompt, images=[image] if image else None)
        append_log(
            settings.log_dir,
            {
                "event": "chat",
                "request_id": request_id or "",
                "session_id": session_id or "",
                "query": query,
                "search_query": search_query,
                "intent_hint": intent_hint,
                "must_filters": must_filters,
                "must_not_filters": must_not_filters,
                "result_count": len(items),
                "graph_anchor_id": anchor_id,
                "graph_used": intent_hint == "graph_pairing" and bool(anchor_id),
                "has_image": bool(image),
                "latency_ms": int((time.perf_counter() - started_at) * 1000),
            },
        )
        return message, items
