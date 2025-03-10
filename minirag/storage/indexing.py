import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from minirag.core.utils import (
    logger,
    compute_mdhash_id,
)
from minirag.core.base import (
    DocStatus,
    DocProcessingStatus,
)
from minirag.core.exceptions import StorageException

from .chunking import chunking_by_token_size
from .extract import extract_entities


async def index_documents(
    documents: List[str],
    minirag_instance,
):
    """
    Index documents into the MiniRAG system
    
    Args:
        documents: List of document strings to index
        minirag_instance: MiniRAG instance
        
    Returns:
        List of document IDs that were indexed
    """
    try:
        logger.info(f"Indexing {len(documents)} documents")
        
        # Create document IDs
        doc_ids = []
        new_docs = {}
        
        for doc in documents:
            doc_id = compute_mdhash_id(doc.strip(), prefix="doc-")
            doc_ids.append(doc_id)
            new_docs[doc_id] = {"content": doc.strip()}
        
        # Filter out documents that already exist
        _add_doc_keys = await minirag_instance.full_docs.filter_keys(list(new_docs.keys()))
        new_docs = {k: v for k, v in new_docs.items() if k in _add_doc_keys}
        
        if not new_docs:
            logger.warning("All documents are already in the storage")
            return []
            
        logger.info(f"Inserting {len(new_docs)} new documents")
        
        # Create document status entries
        now = datetime.now().isoformat()
        doc_status_entries = {}
        
        for doc_id, doc in new_docs.items():
            content = doc["content"]
            doc_status_entries[doc_id] = DocProcessingStatus(
                content=content,
                content_summary=content[:100] + "..." if len(content) > 100 else content,
                content_length=len(content),
                status=DocStatus.PROCESSING,
                created_at=now,
                updated_at=now,
            )
        
        # Update document status
        await minirag_instance.doc_status.upsert(doc_status_entries)
        
        # Chunk documents
        inserting_chunks = {}
        for doc_id, doc in new_docs.items():
            chunks = {
                compute_mdhash_id(dp["content"], prefix="chunk-"): {
                    **dp,
                    "full_doc_id": doc_id,
                }
                for dp in chunking_by_token_size(
                    doc["content"],
                    overlap_token_size=minirag_instance.chunk_overlap_token_size,
                    max_token_size=minirag_instance.chunk_token_size,
                    tiktoken_model=minirag_instance.tiktoken_model_name,
                )
            }
            inserting_chunks.update(chunks)
            
            # Update document status with chunk count
            doc_status_entries[doc_id].chunks_count = len(chunks)
            doc_status_entries[doc_id].status = DocStatus.PROCESSED
            doc_status_entries[doc_id].updated_at = datetime.now().isoformat()
        
        # Filter out chunks that already exist
        _add_chunk_keys = await minirag_instance.text_chunks.filter_keys(
            list(inserting_chunks.keys())
        )
        inserting_chunks = {
            k: v for k, v in inserting_chunks.items() if k in _add_chunk_keys
        }
        
        if not inserting_chunks:
            logger.warning("All chunks are already in the storage")
            return []
            
        logger.info(f"Inserting {len(inserting_chunks)} new chunks")
        
        # Insert chunks into vector database
        await minirag_instance.chunks_vdb.upsert(inserting_chunks)
        
        # Extract entities and relationships
        logger.info("Extracting entities and relationships...")
        maybe_new_kg = await extract_entities(
            inserting_chunks,
            minirag_instance.chunk_entity_relation_graph,
            minirag_instance.entities_vdb,
            minirag_instance.entity_name_vdb,
            minirag_instance.relationships_vdb,
            minirag_instance.global_config,
            minirag_instance.llm_model_func if hasattr(minirag_instance, "llm_model_func") else None,
        )
        
        if maybe_new_kg is None:
            logger.warning("No new entities and relationships found")
        else:
            minirag_instance.chunk_entity_relation_graph = maybe_new_kg
        
        # Update storage
        await minirag_instance.full_docs.upsert(new_docs)
        await minirag_instance.text_chunks.upsert(inserting_chunks)
        await minirag_instance.doc_status.upsert(doc_status_entries)
        
        # Commit changes
        await minirag_instance._index_done()
        
        return list(new_docs.keys())
        
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        
        # Update document status to failed
        if 'doc_status_entries' in locals() and 'doc_id' in locals():
            for doc_id in doc_ids:
                if doc_id in doc_status_entries:
                    doc_status_entries[doc_id].status = DocStatus.FAILED
                    doc_status_entries[doc_id].error = str(e)
                    doc_status_entries[doc_id].updated_at = datetime.now().isoformat()
            
            await minirag_instance.doc_status.upsert(doc_status_entries)
            
        raise StorageException(f"Failed to index documents: {str(e)}") 