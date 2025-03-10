import asyncio
from typing import Dict, List, Any, Optional
import numpy as np

from minirag.core.utils import (
    logger,
)
from minirag.core.base import (
    QueryParam,
)
from minirag.core.exceptions import QueryException


async def retrieve_relevant_chunks(
    query: str,
    chunks_vdb,
    text_chunks,
    param: QueryParam,
):
    """
    Retrieve relevant chunks for a query
    
    Args:
        query: Query string
        chunks_vdb: Chunks vector database instance
        text_chunks: Text chunks storage instance
        param: Query parameters
        
    Returns:
        List of relevant chunks
    """
    try:
        # Query the vector database
        results = await chunks_vdb.query(query, param.top_k)
        
        if not results:
            logger.warning(f"No relevant chunks found for query: {query}")
            return []
            
        # Get the full chunk data
        chunk_ids = [result["id"] for result in results]
        chunks = await text_chunks.get_by_ids(chunk_ids)
        
        # Combine the results
        for i, result in enumerate(results):
            if chunks[i] is not None:
                result.update(chunks[i])
                
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving relevant chunks: {str(e)}")
        raise QueryException(f"Failed to retrieve relevant chunks: {str(e)}")


async def retrieve_relevant_entities(
    query: str,
    entity_vdb,
    entity_name_vdb,
    param: QueryParam,
):
    """
    Retrieve relevant entities for a query
    
    Args:
        query: Query string
        entity_vdb: Entity vector database instance
        entity_name_vdb: Entity name vector database instance
        param: Query parameters
        
    Returns:
        List of relevant entities
    """
    try:
        # Query both entity databases
        entity_results = await entity_vdb.query(query, param.top_k)
        entity_name_results = await entity_name_vdb.query(query, param.top_k)
        
        # Combine and deduplicate results
        combined_results = {}
        
        for result in entity_results + entity_name_results:
            entity_id = result["id"]
            if entity_id not in combined_results or result["distance"] > combined_results[entity_id]["distance"]:
                combined_results[entity_id] = result
                
        # Sort by distance
        sorted_results = sorted(
            combined_results.values(),
            key=lambda x: x["distance"],
            reverse=True
        )
        
        return sorted_results[:param.top_k]
        
    except Exception as e:
        logger.error(f"Error retrieving relevant entities: {str(e)}")
        raise QueryException(f"Failed to retrieve relevant entities: {str(e)}")


async def retrieve_entity_relationships(
    entity_ids: List[str],
    relationships_vdb,
    graph_storage,
    param: QueryParam,
):
    """
    Retrieve relationships for entities
    
    Args:
        entity_ids: List of entity IDs
        relationships_vdb: Relationships vector database instance
        graph_storage: Graph storage instance
        param: Query parameters
        
    Returns:
        List of relationships
    """
    try:
        relationships = []
        
        for entity_id in entity_ids:
            # Get edges from the graph
            edges = await graph_storage.get_node_edges(entity_id)
            
            if not edges:
                continue
                
            for source, target in edges:
                # Get the edge data
                edge_data = await graph_storage.get_edge(source, target)
                
                if edge_data:
                    relationships.append({
                        "source": source,
                        "target": target,
                        "data": edge_data,
                    })
                    
        return relationships
        
    except Exception as e:
        logger.error(f"Error retrieving entity relationships: {str(e)}")
        raise QueryException(f"Failed to retrieve entity relationships: {str(e)}")


async def generate_query_context(
    query: str,
    chunks: List[Dict[str, Any]],
    entities: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    param: QueryParam,
):
    """
    Generate context for a query
    
    Args:
        query: Query string
        chunks: List of relevant chunks
        entities: List of relevant entities
        relationships: List of relevant relationships
        param: Query parameters
        
    Returns:
        Query context dictionary
    """
    try:
        # Format chunks
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            formatted_chunks.append({
                "content": chunk["content"],
                "distance": chunk.get("distance", 0),
                "index": i + 1,
            })
            
        # Format entities
        formatted_entities = []
        for i, entity in enumerate(entities):
            formatted_entities.append({
                "name": entity.get("entity_name", "Unknown"),
                "type": entity.get("entity_type", "Unknown"),
                "content": entity.get("content", ""),
                "distance": entity.get("distance", 0),
                "index": i + 1,
            })
            
        # Format relationships
        formatted_relationships = []
        for i, rel in enumerate(relationships):
            formatted_relationships.append({
                "source": rel["source"],
                "target": rel["target"],
                "type": rel["data"].get("rel_type", "Unknown"),
                "content": rel["data"].get("content", ""),
                "index": i + 1,
            })
            
        # Create context
        context = {
            "query": query,
            "chunks": formatted_chunks,
            "entities": formatted_entities,
            "relationships": formatted_relationships,
        }
        
        return context
        
    except Exception as e:
        logger.error(f"Error generating query context: {str(e)}")
        raise QueryException(f"Failed to generate query context: {str(e)}")


async def minirag_query(
    query: str,
    graph_storage,
    entity_vdb,
    entity_name_vdb,
    relationships_vdb,
    chunks_vdb,
    text_chunks,
    embedding_func,
    param: QueryParam,
    global_config: Dict[str, Any],
):
    """
    Execute a MiniRAG query
    
    Args:
        query: Query string
        graph_storage: Graph storage instance
        entity_vdb: Entity vector database instance
        entity_name_vdb: Entity name vector database instance
        relationships_vdb: Relationships vector database instance
        chunks_vdb: Chunks vector database instance
        text_chunks: Text chunks storage instance
        embedding_func: Embedding function
        param: Query parameters
        global_config: Global configuration
        
    Returns:
        Query result
    """
    try:
        logger.info(f"Executing MiniRAG query: {query}")
        
        # Retrieve relevant chunks
        chunks = await retrieve_relevant_chunks(
            query,
            chunks_vdb,
            text_chunks,
            param,
        )
        
        # Retrieve relevant entities
        entities = await retrieve_relevant_entities(
            query,
            entity_vdb,
            entity_name_vdb,
            param,
        )
        
        # Retrieve entity relationships
        entity_ids = [entity["id"] for entity in entities]
        relationships = await retrieve_entity_relationships(
            entity_ids,
            relationships_vdb,
            graph_storage,
            param,
        )
        
        # Generate context
        context = await generate_query_context(
            query,
            chunks,
            entities,
            relationships,
            param,
        )
        
        # If only context is needed, return it
        if param.only_need_context:
            return context
            
        # Generate prompt (would be implemented in a real system)
        # For now, just return the context
        return context
        
    except Exception as e:
        logger.error(f"Error executing MiniRAG query: {str(e)}")
        raise QueryException(f"Failed to execute MiniRAG query: {str(e)}") 