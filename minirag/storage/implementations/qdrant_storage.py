import asyncio
import os
from dataclasses import dataclass
import numpy as np
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from minirag.core.utils import (
    logger,
    compute_mdhash_id,
)

from minirag.core.base import (
    BaseVectorStorage,
)
from minirag.core.exceptions import StorageException


@dataclass
class QdrantStorage(BaseVectorStorage):
    similarity_threshold: float = 0.2
    
    def __post_init__(self):
        # Use global config value if specified, otherwise use default
        config = self.global_config.get("vector_db_storage_cls_kwargs", {})
        self.similarity_threshold = config.get(
            "similarity_threshold", self.similarity_threshold
        )
        
        # Initialize Qdrant client
        self.host = config.get("host", "localhost")
        self.port = config.get("port", 6333)
        self.grpc_port = config.get("grpc_port", 6334)
        self.prefer_grpc = config.get("prefer_grpc", False)
        self.api_key = config.get("api_key", None)
        self.https = config.get("https", False)
        self.collection_name = f"minirag_{self.namespace}"
        
        try:
            self._client = QdrantClient(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                prefer_grpc=self.prefer_grpc,
                api_key=self.api_key,
                https=self.https
            )
            
            # Check if collection exists, if not create it
            collections = self._client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_func.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created new Qdrant collection: {self.collection_name}")
            else:
                logger.info(f"Using existing Qdrant collection: {self.collection_name}")
                
        except Exception as e:
            raise StorageException(f"Failed to initialize Qdrant client: {str(e)}")

    async def upsert(self, data: dict[str, dict]):
        """
        Insert or update vectors in Qdrant
        
        Args:
            data: Dictionary of ID to document data
        """
        logger.info(f"Inserting {len(data)} vectors to {self.namespace}")
        if not len(data):
            logger.warning("You inserted empty data to vector DB")
            return []
            
        try:
            # Prepare data for batch insertion
            ids = []
            contents = []
            payloads = []
            
            for key, value in data.items():
                ids.append(key)
                contents.append(value["content"])
                
                # Prepare payload with metadata
                payload = {k: v for k, v in value.items() if k in self.meta_fields or k == "content"}
                payload["id"] = key
                payloads.append(payload)
            
            # Generate embeddings in batches
            embeddings = await self.embedding_func(contents)
            
            # Create points for Qdrant
            points = []
            for i in range(len(ids)):
                points.append(
                    models.PointStruct(
                        id=ids[i],
                        vector=embeddings[i].tolist(),
                        payload=payloads[i]
                    )
                )
            
            # Upsert points to Qdrant
            self._client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            logger.info(f"Successfully inserted {len(points)} vectors to {self.namespace}")
            return ids
            
        except Exception as e:
            logger.error(f"Error inserting vectors to {self.namespace}: {str(e)}")
            raise StorageException(f"Failed to upsert vectors: {str(e)}")

    async def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Query vectors from Qdrant
        
        Args:
            query: Query string
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with query results
        """
        try:
            # Generate embedding for query
            embedding = await self.embedding_func([query])
            embedding = embedding[0].tolist()
            
            # Search in Qdrant
            search_result = self._client.search(
                collection_name=self.collection_name,
                query_vector=embedding,
                limit=top_k,
                score_threshold=self.similarity_threshold
            )
            
            # Format results
            results = []
            for hit in search_result:
                result = {
                    **hit.payload,
                    "id": hit.id,
                    "distance": hit.score,
                }
                results.append(result)
                
            return results
            
        except Exception as e:
            logger.error(f"Error querying vectors from {self.namespace}: {str(e)}")
            raise StorageException(f"Failed to query vectors: {str(e)}")

    async def delete(self, ids: List[str]):
        """
        Delete vectors from Qdrant
        
        Args:
            ids: List of vector IDs to delete
        """
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=ids
                )
            )
            logger.info(f"Successfully deleted {len(ids)} vectors from {self.namespace}")
        except Exception as e:
            logger.error(f"Error deleting vectors from {self.namespace}: {str(e)}")
            raise StorageException(f"Failed to delete vectors: {str(e)}")

    async def delete_entity(self, entity_name: str):
        """
        Delete an entity by name
        
        Args:
            entity_name: Name of the entity to delete
        """
        try:
            entity_id = compute_mdhash_id(entity_name, prefix="ent-")
            logger.debug(f"Attempting to delete entity {entity_name} with ID {entity_id}")
            
            # Delete the entity
            await self.delete([entity_id])
            logger.debug(f"Successfully deleted entity {entity_name}")
        except Exception as e:
            logger.error(f"Error deleting entity {entity_name}: {str(e)}")
            raise StorageException(f"Failed to delete entity: {str(e)}")

    async def index_done_callback(self):
        """No special action needed for Qdrant after indexing"""
        pass 