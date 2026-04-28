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

        if must_filters:
            for key, value in must_filters.items():
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    values = [str(v).strip() for v in value if str(v).strip()]
                    if values:
                        must_conditions.append(
                            models.FieldCondition(key=key, match=models.MatchAny(any=values))
                        )
                    continue
                value_str = str(value).strip()
                if not value_str:
                    continue
                must_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value_str))
                )

        if must_not_filters:
            for key, value in must_not_filters.items():
                if value is None:
                    continue
                if isinstance(value, (list, tuple)):
                    values = [str(v).strip() for v in value if str(v).strip()]
                    if values:
                        must_not_conditions.append(
                            models.FieldCondition(key=key, match=models.MatchAny(any=values))
                        )
                    continue
                value_str = str(value).strip()
                if not value_str:
                    continue
                must_not_conditions.append(
                    models.FieldCondition(key=key, match=models.MatchValue(value=value_str))
                )

        if not must_conditions and not must_not_conditions:
            return None
        return models.Filter(must=must_conditions or None, must_not=must_not_conditions or None)

    def query(
        self,
        query_vector: List[float],
        limit: int,
        must_filters: Dict[str, str] | None = None,
        must_not_filters: Dict[str, List[str] | str] | None = None,
    ):
        query_filter = self._build_filter(must_filters, must_not_filters)

        if hasattr(self.client, "query_points"):
            try:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=limit,
                    query_filter=query_filter,
                )
            except TypeError:
                response = self.client.query_points(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    query_filter=query_filter,
                )
            return response.points if hasattr(response, "points") else response

        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
        )

    def retrieve_by_article_ids(self, article_ids: List[str]):
        if not article_ids:
            return []

        point_ids = parse_numeric_ids(article_ids)
        if not point_ids:
            return []

        return self.client.retrieve(collection_name=self.collection_name, ids=point_ids)
