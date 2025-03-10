import asyncio
from typing import Dict, List, Optional, Any
import re

from minirag.core.utils import (
    logger,
    compute_mdhash_id,
)
from minirag.core.exceptions import EntityExtractionException


async def extract_entities(
    chunks: Dict[str, Dict[str, Any]],
    knowledge_graph_inst,
    entity_vdb,
    entity_name_vdb,
    relationships_vdb,
    global_config: Dict[str, Any],
    llm_func=None,
):
    """
    Extract entities and relationships from text chunks
    
    Args:
        chunks: Dictionary of chunk IDs to chunk data
        knowledge_graph_inst: Knowledge graph storage instance
        entity_vdb: Entity vector database instance
        entity_name_vdb: Entity name vector database instance
        relationships_vdb: Relationships vector database instance
        global_config: Global configuration dictionary
        llm_func: LLM function for entity extraction (optional)
        
    Returns:
        Updated knowledge graph instance
    """
    try:
        logger.info(f"Extracting entities from {len(chunks)} chunks")
        
        # If no LLM function is provided, use a simple regex-based approach
        if llm_func is None:
            return await simple_entity_extraction(
                chunks,
                knowledge_graph_inst,
                entity_vdb,
                entity_name_vdb,
                relationships_vdb
            )
        
        # Otherwise, use the LLM for entity extraction
        # This would be implemented in a more sophisticated way
        # but for simplicity, we'll use the simple approach
        return await simple_entity_extraction(
            chunks,
            knowledge_graph_inst,
            entity_vdb,
            entity_name_vdb,
            relationships_vdb
        )
        
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        raise EntityExtractionException(f"Failed to extract entities: {str(e)}")


async def simple_entity_extraction(
    chunks: Dict[str, Dict[str, Any]],
    knowledge_graph_inst,
    entity_vdb,
    entity_name_vdb,
    relationships_vdb,
):
    """
    Simple entity extraction using regex patterns
    
    Args:
        chunks: Dictionary of chunk IDs to chunk data
        knowledge_graph_inst: Knowledge graph storage instance
        entity_vdb: Entity vector database instance
        entity_name_vdb: Entity name vector database instance
        relationships_vdb: Relationships vector database instance
        
    Returns:
        Updated knowledge graph instance
    """
    # Simple patterns for entity extraction
    # In a real implementation, this would be much more sophisticated
    name_pattern = r'(?:[A-Z][a-z]+ )+[A-Z][a-z]+'
    org_pattern = r'(?:[A-Z][a-z]* )*(?:Inc\.|Corp\.|LLC|Ltd\.|Company|Organization)'
    location_pattern = r'(?:[A-Z][a-z]+ )+(?:City|Town|Country|State|Province|Region)'
    
    entities = {}
    relationships = {}
    
    for chunk_id, chunk_data in chunks.items():
        content = chunk_data["content"]
        
        # Extract entities
        names = re.findall(name_pattern, content)
        orgs = re.findall(org_pattern, content)
        locations = re.findall(location_pattern, content)
        
        # Process names
        for name in names:
            entity_id = compute_mdhash_id(name, prefix="ent-")
            if entity_id not in entities:
                entities[entity_id] = {
                    "entity_name": name,
                    "entity_type": "PERSON",
                    "content": f"Person: {name}",
                }
                
        # Process organizations
        for org in orgs:
            entity_id = compute_mdhash_id(org, prefix="ent-")
            if entity_id not in entities:
                entities[entity_id] = {
                    "entity_name": org,
                    "entity_type": "ORGANIZATION",
                    "content": f"Organization: {org}",
                }
                
        # Process locations
        for location in locations:
            entity_id = compute_mdhash_id(location, prefix="ent-")
            if entity_id not in entities:
                entities[entity_id] = {
                    "entity_name": location,
                    "entity_type": "LOCATION",
                    "content": f"Location: {location}",
                }
                
        # Create relationships between entities and chunks
        for entity_id in entities:
            rel_id = compute_mdhash_id(f"{entity_id}_{chunk_id}", prefix="rel-")
            if rel_id not in relationships:
                relationships[rel_id] = {
                    "src_id": entity_id,
                    "tgt_id": chunk_id,
                    "rel_type": "MENTIONED_IN",
                    "content": f"Entity {entities[entity_id]['entity_name']} is mentioned in chunk {chunk_id}",
                }
    
    # Update the knowledge graph
    for entity_id, entity_data in entities.items():
        await knowledge_graph_inst.upsert_node(entity_id, entity_data)
        
    # Update the entity vector databases
    if entities:
        await entity_vdb.upsert(entities)
        await entity_name_vdb.upsert(entities)
        
    # Update relationships
    for rel_id, rel_data in relationships.items():
        src_id = rel_data["src_id"]
        tgt_id = rel_data["tgt_id"]
        await knowledge_graph_inst.upsert_edge(src_id, tgt_id, rel_data)
        
    # Update the relationships vector database
    if relationships:
        await relationships_vdb.upsert(relationships)
        
    logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
    return knowledge_graph_inst 