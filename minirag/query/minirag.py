import asyncio
from typing import Dict, List, Any, Optional, Union
import os
from dataclasses import asdict, dataclass, field

from minirag.core.utils import (
    logger,
    EmbeddingFunc,
    compute_mdhash_id,
)
from minirag.core.base import (
    QueryParam,
    StorageNameSpace,
    BaseGraphStorage,
    BaseKVStorage,
    BaseVectorStorage,
    DocStatusStorage,
)
from minirag.core.exceptions import (
    ConfigException,
    StorageException,
)

from .operates import minirag_query


@dataclass
class MiniRAG:
    """MiniRAG main class for document retrieval and querying"""
    
    # Storage configuration
    working_dir: str
    embedding_func: EmbeddingFunc
    
    # Storage instances
    full_docs: BaseKVStorage = None
    text_chunks: BaseKVStorage = None
    doc_status: DocStatusStorage = None
    chunks_vdb: BaseVectorStorage = None
    entities_vdb: BaseVectorStorage = None
    entity_name_vdb: BaseVectorStorage = None
    relationships_vdb: BaseVectorStorage = None
    chunk_entity_relation_graph: BaseGraphStorage = None
    
    # Chunking configuration
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100
    tiktoken_model_name: str = "gpt-4o"
    
    # Global configuration
    global_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize MiniRAG with default configuration if not provided"""
        # Set up working directory
        os.makedirs(self.working_dir, exist_ok=True)
        
        # Set up global configuration
        if not self.global_config:
            self.global_config = {
                "working_dir": self.working_dir,
                "embedding_batch_num": 32,
                "vector_db_storage_cls_kwargs": {
                    "similarity_threshold": 0.2,
                },
            }
        else:
            self.global_config["working_dir"] = self.working_dir
            
        # Validate required storage instances
        if not all([
            self.full_docs,
            self.text_chunks,
            self.doc_status,
            self.chunks_vdb,
            self.entities_vdb,
            self.entity_name_vdb,
            self.relationships_vdb,
            self.chunk_entity_relation_graph,
        ]):
            raise ConfigException("All storage instances must be provided")
            
        logger.info("MiniRAG initialized successfully")
    
    async def _index_done(self):
        """Commit all storage operations after indexing"""
        await self.full_docs.index_done_callback()
        await self.text_chunks.index_done_callback()
        await self.doc_status.index_done_callback()
        await self.chunks_vdb.index_done_callback()
        await self.entities_vdb.index_done_callback()
        await self.entity_name_vdb.index_done_callback()
        await self.relationships_vdb.index_done_callback()
        await self.chunk_entity_relation_graph.index_done_callback()
        
    async def _query_done(self):
        """Commit all storage operations after querying"""
        await self.full_docs.query_done_callback()
        await self.text_chunks.query_done_callback()
        await self.doc_status.query_done_callback()
        await self.chunks_vdb.query_done_callback()
        await self.entities_vdb.query_done_callback()
        await self.entity_name_vdb.query_done_callback()
        await self.relationships_vdb.query_done_callback()
        await self.chunk_entity_relation_graph.query_done_callback()
    
    async def index(self, documents: List[str]) -> List[str]:
        """
        Index documents into MiniRAG
        
        Args:
            documents: List of document strings to index
            
        Returns:
            List of document IDs that were indexed
        """
        from minirag.storage.indexing import index_documents
        
        return await index_documents(documents, self)
    
    async def query(self, query: str, param: Optional[QueryParam] = None) -> Dict[str, Any]:
        """
        Query MiniRAG for information
        
        Args:
            query: Query string
            param: Query parameters (optional)
            
        Returns:
            Query result
        """
        if param is None:
            param = QueryParam()
            
        try:
            result = await minirag_query(
                query,
                self.chunk_entity_relation_graph,
                self.entities_vdb,
                self.entity_name_vdb,
                self.relationships_vdb,
                self.chunks_vdb,
                self.text_chunks,
                self.embedding_func,
                param,
                self.global_config,
            )
            
            await self._query_done()
            return result
            
        except Exception as e:
            logger.error(f"Error querying MiniRAG: {str(e)}")
            raise StorageException(f"Failed to query MiniRAG: {str(e)}")
    
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from MiniRAG
        
        Args:
            doc_id: Document ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get document chunks
            doc = await self.full_docs.get_by_id(doc_id)
            if not doc:
                logger.warning(f"Document {doc_id} not found")
                return False
                
            # Get all chunks for this document
            all_chunks = await self.text_chunks.filter(
                lambda x: x.get("full_doc_id") == doc_id
            )
            
            # Delete chunks from vector database
            chunk_ids = list(all_chunks.keys())
            if chunk_ids:
                await self.chunks_vdb.delete(chunk_ids)
                
            # Delete document and chunks from KV storage
            await self.full_docs.delete([doc_id])
            await self.text_chunks.delete(chunk_ids)
            await self.doc_status.delete([doc_id])
            
            # Commit changes
            await self._index_done()
            
            logger.info(f"Document {doc_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {str(e)}")
            return False 