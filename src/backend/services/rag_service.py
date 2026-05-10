from __future__ import annotations

from typing import Any, Dict, List

from PIL import Image

from src.backend.core.utils import normalize_text
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

    async def chat(self, query: str, image: Image.Image | None = None) -> tuple[str, List[dict]]:
        analysis = self.llm.analyze_user_query(query)
        search_query = str(analysis.get("search_query_en", "") or "").strip() or query
        must_filters = self._validate_filters(analysis.get("must_filters", {}))
        must_not_filters = self._validate_must_not(analysis.get("must_not_filters", {}))
        dense_vec, sparse_idx, sparse_val = self.embedder.encode_hybrid(search_query, search_query, image=image)
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
                "image_path": str(payload.get("image_path", "")),
                "price": str(payload.get("price", "") or ""),
            })

        context = self._format_context(points)
        if not context:
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
        return message, items
