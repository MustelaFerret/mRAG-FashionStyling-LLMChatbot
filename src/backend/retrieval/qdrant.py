from __future__ import annotations

from typing import Dict, List

from qdrant_client import QdrantClient, models

from src.backend.core.utils import parse_numeric_ids


class QdrantStore:
    def __init__(self, db_path: str, collection_name: str):
        self.collection_name = collection_name
        self.client = QdrantClient(path=db_path)

    def _build_filter(
        self,
        must_filters: Dict[str, str] | None,
        must_not_filters: Dict[str, List[str] | str] | None,
    ) -> models.Filter | None:
        must_conditions: List[models.FieldCondition] = []
        must_not_conditions: List[models.FieldCondition] = []
        key_aliases = {
            "product_type": ["product_type", "product_type_name"],
            "colour_group": ["colour_group", "colour_group_name"],
        }

        def _build_match(value):
            if isinstance(value, (list, tuple)):
                values = [str(v).strip() for v in value if str(v).strip()]
                if not values:
                    return None
                return models.MatchAny(any=values)
            value_str = str(value).strip()
            if not value_str:
                return None
            return models.MatchValue(value=value_str)

        def _build_conditions_for_key(key: str, match):
            keys = key_aliases.get(key, [key])
            return [models.FieldCondition(key=k, match=match) for k in keys]

        if must_filters:
            for key, value in must_filters.items():
                if value is None:
                    continue
                match = _build_match(value)
                if match is None:
                    continue
                conditions = _build_conditions_for_key(key, match)
                if len(conditions) == 1:
                    must_conditions.append(conditions[0])
                else:
                    must_conditions.append(models.Filter(should=conditions))

        if must_not_filters:
            for key, value in must_not_filters.items():
                if value is None:
                    continue
                match = _build_match(value)
                if match is None:
                    continue
                conditions = _build_conditions_for_key(key, match)
                if len(conditions) == 1:
                    must_not_conditions.append(conditions[0])
                else:
                    must_not_conditions.append(models.Filter(should=conditions))

        if not must_conditions and not must_not_conditions:
            return None
        return models.Filter(must=must_conditions or None, must_not=must_not_conditions or None)

    def query(
        self,
        query_vector: List[float],
        limit: int,
        must_filters: Dict[str, str] | None = None,
        must_not_filters: Dict[str, List[str] | str] | None = None,
        vector_name: str | None = "dense",
    ):
        query_filter = self._build_filter(must_filters, must_not_filters)

        if hasattr(self.client, "query_points"):
            try:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=limit,
                    query_filter=query_filter,
                    vector_name=vector_name,
                )
            except (TypeError, Exception):
                try:
                    response = self.client.query_points(
                        collection_name=self.collection_name,
                        query_vector=query_vector,
                        limit=limit,
                        query_filter=query_filter,
                        vector_name=vector_name,
                    )
                except (TypeError, Exception):
                    response = self.client.query_points(
                        collection_name=self.collection_name,
                        query_vector=query_vector,
                        limit=limit,
                        query_filter=query_filter,
                    )
            return response.points if hasattr(response, "points") else response

        try:
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
                vector_name=vector_name,
            )
        except (TypeError, Exception):
            return self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=limit,
            )

    def ensure_collection(
        self,
        dense_name: str = "dense",
        sparse_name: str = "sparse",
        size: int = 768,
        distance: models.Distance = models.Distance.COSINE,
        reset: bool = False,
        payload_index_fields: List[str] | None = None,
    ) -> None:
        if reset and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={dense_name: models.VectorParams(size=size, distance=distance)},
                sparse_vectors_config={sparse_name: models.SparseVectorParams()},
            )

        for field in payload_index_fields or []:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                pass

    def retrieve_by_article_ids(self, article_ids: List[str]):
        if not article_ids:
            return []

        point_ids = parse_numeric_ids(article_ids)
        if not point_ids:
            return []

        return self.client.retrieve(collection_name=self.collection_name, ids=point_ids)

    def hybrid_search(
        self,
        dense_vector: List[float],
        sparse_indices: List[int],
        sparse_values: List[float],
        limit: int = 10,
        must_filters: Dict[str, str] | None = None,
        must_not_filters: Dict[str, List[str] | str] | None = None,
    ):
        query_filter = self._build_filter(must_filters, must_not_filters)
        prefetch = [
            models.Prefetch(
                query=dense_vector,
                using="dense",
                limit=limit * 3,
                filter=query_filter,
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                using="sparse",
                limit=limit * 3,
                filter=query_filter,
            ),
        ]
        response = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )
        return response.points if hasattr(response, "points") else response
