from __future__ import annotations

from typing import Dict, List

from qdrant_client import QdrantClient, models

from src.backend.core.utils import parse_numeric_ids


class QdrantStore:
    def __init__(self, db_path: str, collection_name: str):
        self.collection_name = collection_name
        self.client = QdrantClient(path=db_path)

    def _build_filter(self, constraints: Dict[str, str] | None) -> models.Filter | None:
        if not constraints:
            return None

        must_conditions: List[models.FieldCondition] = []
        for key, value in constraints.items():
            if value is None:
                continue
            value_str = str(value).strip()
            if not value_str:
                continue
            must_conditions.append(
                models.FieldCondition(key=key, match=models.MatchValue(value=value_str))
            )

        if not must_conditions:
            return None
        return models.Filter(must=must_conditions)

    def query(self, query_vector: List[float], limit: int, constraints: Dict[str, str] | None = None):
        query_filter = self._build_filter(constraints)

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
