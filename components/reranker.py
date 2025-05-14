from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.retrievers.base import RetrieverResultItem
import neo4j
from flashrank import Ranker, RerankRequest
from pydantic import ValidationError

from neo4j_graphrag.exceptions import (
    EmbeddingRequiredError,
    SearchValidationError,
)
from neo4j_graphrag.neo4j_queries import get_search_query
from neo4j_graphrag.types import (
    RawSearchResult,
    RetrieverResultItem,
    SearchType,
    VectorCypherSearchModel,
)
from neo4j_graphrag.types import RawSearchResult, RetrieverResult, RetrieverResultItem
from typing import Any, Optional


class RerankableRetriever(VectorCypherRetriever):
    """
    A retriever that can be reranked based on the scores of the retrieved items.
    This is a subclass of VectorCypherRetriever.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_search_results(
        self,
        query_vector: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = 5,
        effective_search_ratio: int = 1,
        query_params: Optional[dict[str, Any]] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> RawSearchResult:
        """Get the top_k nearest neighbor embeddings for either provided query_vector or query_text.
        See the following documentation for more details:

        - `Query a vector index <https://neo4j.com/docs/cypher-manual/current/indexes-for-vector-search/#indexes-vector-query>`_
        - `db.index.vector.queryNodes() <https://neo4j.com/docs/operations-manual/5/reference/procedures/#procedure_db_index_vector_queryNodes>`_

        To query by text, an embedder must be provided when the class is instantiated.  The embedder is not required if `query_vector` is passed.

        Args:
            query_vector (Optional[list[float]]): The vector embeddings to get the closest neighbors of. Defaults to None.
            query_text (Optional[str]): The text to get the closest neighbors of. Defaults to None.
            top_k (int): The number of neighbors to return. Defaults to 5.
            effective_search_ratio (int): Controls the candidate pool size by multiplying top_k to balance query accuracy and performance.
                Defaults to 1.
            query_params (Optional[dict[str, Any]]): Parameters for the Cypher query. Defaults to None.
            filters (Optional[dict[str, Any]]): Filters for metadata pre-filtering. Defaults to None.

        Raises:
            SearchValidationError: If validation of the input arguments fail.
            EmbeddingRequiredError: If no embedder is provided.

        Returns:
            RawSearchResult: The results of the search query as a list of neo4j.Record and an optional metadata dict
        """
        try:
            validated_data = VectorCypherSearchModel(
                query_vector=query_vector,
                query_text=query_text,
                top_k=top_k,
                effective_search_ratio=effective_search_ratio,
                query_params=query_params,
                filters=filters,
            )
        except ValidationError as e:
            raise SearchValidationError(e.errors()) from e

        parameters = validated_data.model_dump(exclude_none=True)
        parameters["vector_index_name"] = self.index_name
        if filters:
            del parameters["filters"]

        if query_text:
            if not self.embedder:
                raise EmbeddingRequiredError(
                    "Embedding method required for text query."
                )
            parameters["query_vector"] = self.embedder.embed_query(query_text)
            del parameters["query_text"]

        if query_params:
            for key, value in query_params.items():
                if key not in parameters:
                    parameters[key] = value
            del parameters["query_params"]

        search_query, search_params = get_search_query(
            search_type=SearchType.VECTOR,
            retrieval_query=self.retrieval_query,
            node_label=self._node_label,
            embedding_node_property=self._node_embedding_property,
            embedding_dimension=self._embedding_dimension,
            filters=filters,
        )
        parameters.update(search_params)

        records, _, _ = self.driver.execute_query(
            search_query,
            parameters,
            database_=self.neo4j_database,
            routing_=neo4j.RoutingControl.READ,
        )
        return RawSearchResult(
            records=records,
        ), query_text
    
    def rerank(self, items: list[RetrieverResultItem], query: str) -> list[RetrieverResultItem]:
        """
        Rerank the retrieved items based on the scores of the retrieved items.
        This is a simple reranking function that sorts the items based on their scores.
        Args:
            items (list[RetrieverResultItem]): The retrieved items.
            query (str): The query used to retrieve the items.
        Returns:
            list[RetrieverResultItem]: The reranked items.
        """
        ranker = Ranker(max_length=256) #token sized based on largest query + extra space for retrieved items
        rerankrequest = RerankRequest(query=query, passages=items)
        results = ranker.rerank(rerankrequest)
        return results
    
    def format_sentences_for_flashrank(self, items):
        passages = []
        idx = 1
        for item in items:
            sentences = [s.strip() for s in item.content.split(',') if s.strip()]
            for sentence in sentences:
                passages.append({
                    "id": idx,
                    "text": sentence,
                    "meta": {
                        "startnode": item.metadata.get("startnode", ""),
                        "similarity_score": item.metadata.get("similarity_score", 0.0),
                    }
                })
                idx += 1
        return passages
    
    def format_reranked_items(self, items) -> RetrieverResultItem:
        """
        Format the reranked items into a list of RetrieverResultItem.
        Args:
            items (list[RetrieverResultItem]): The reranked items.
        Returns:
            list[RetrieverResultItem]: The formatted reranked items.
        """
        formatted_items = []
        for item in items:
            formatted_items.append(RetrieverResultItem(
                content=item["text"],
                metadata={
                    "startnode": item["meta"]["startnode"],
                    "similarity_score": item["meta"]["similarity_score"],
                }
            ))
        return formatted_items
    
    def search(self, *args: Any, **kwargs: Any) -> RetrieverResult:
        """Search method. Call the `get_search_results` method that returns
        a list of `neo4j.Record`, and format them using the function returned by
        `get_result_formatter` to return `RetrieverResult`.
        """
        raw_result, query_text = self.get_search_results(*args, **kwargs)
        formatter = self.get_result_formatter()
        search_items = [formatter(record) for record in raw_result.records]
        passages = self.format_sentences_for_flashrank(search_items)
        reranked_items = self.rerank(passages, query_text)
        # format the reranked items
        reranked_items = self.format_reranked_items(reranked_items)
        metadata = raw_result.metadata or {}
        metadata["__retriever"] = self.__class__.__name__
        return RetrieverResult(
            items=reranked_items,
            metadata=metadata,
        )

