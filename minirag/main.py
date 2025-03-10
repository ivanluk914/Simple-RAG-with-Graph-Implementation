import asyncio
import os
import argparse
import logging
from typing import List, Dict, Any, Optional
import numpy as np

from minirag.core.utils import (
    EmbeddingFunc,
    set_logger,
    logger,
)
from minirag.core.base import (
    QueryParam,
)
from minirag.query import MiniRAG
from minirag.storage import (
    JsonKVStorage,
    NetworkXStorage,
    QdrantStorage,
    JsonDocStatusStorage,
)


async def create_embedding_func(model_name: str = "openai") -> EmbeddingFunc:
    """
    Create an embedding function based on the model name
    
    Args:
        model_name: Name of the embedding model to use
        
    Returns:
        EmbeddingFunc instance
    """
    if model_name == "openai":
        try:
            from openai import AsyncOpenAI
            import os
            
            # Check for API key
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
                
            client = AsyncOpenAI(api_key=api_key)
            
            async def openai_embedding_func(texts: List[str]) -> np.ndarray:
                response = await client.embeddings.create(
                    input=texts,
                    model="text-embedding-3-small"
                )
                
                embeddings = [np.array(item.embedding) for item in response.data]
                return np.array(embeddings)
                
            return EmbeddingFunc(
                embedding_dim=1536,  # text-embedding-3-small dimension
                max_token_size=8191,  # text-embedding-3-small token limit
                func=openai_embedding_func,
            )
            
        except ImportError:
            logger.error("OpenAI package not installed. Install with 'pip install openai'")
            raise
            
    elif model_name == "sentence-transformers":
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            async def st_embedding_func(texts: List[str]) -> np.ndarray:
                # SentenceTransformer is not async, so we run it in a separate thread
                embeddings = await asyncio.to_thread(model.encode, texts)
                return embeddings
                
            return EmbeddingFunc(
                embedding_dim=384,  # all-MiniLM-L6-v2 dimension
                max_token_size=256,  # Approximate token limit
                func=st_embedding_func,
            )
            
        except ImportError:
            logger.error("Sentence-Transformers package not installed. Install with 'pip install sentence-transformers'")
            raise
            
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")


async def create_minirag(
    working_dir: str,
    embedding_model: str = "openai",
) -> MiniRAG:
    """
    Create a MiniRAG instance with default storage implementations
    
    Args:
        working_dir: Working directory for storage
        embedding_model: Name of the embedding model to use
        
    Returns:
        MiniRAG instance
    """
    # Create embedding function
    embedding_func = await create_embedding_func(embedding_model)
    
    # Create global config
    global_config = {
        "working_dir": working_dir,
        "embedding_batch_num": 32,
        "vector_db_storage_cls_kwargs": {
            "similarity_threshold": 0.2,
        },
    }
    
    # Create storage instances
    full_docs = JsonKVStorage(
        namespace="full_docs",
        global_config=global_config,
    )
    
    text_chunks = JsonKVStorage(
        namespace="text_chunks",
        global_config=global_config,
    )
    
    doc_status = JsonDocStatusStorage(
        namespace="doc_status",
        global_config=global_config,
        embedding_func=embedding_func,
    )
    
    chunks_vdb = QdrantStorage(
        namespace="chunks",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"full_doc_id", "chunk_order_index"},
    )
    
    entities_vdb = QdrantStorage(
        namespace="entities",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"entity_name", "entity_type"},
    )
    
    entity_name_vdb = QdrantStorage(
        namespace="entity_names",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"entity_name", "entity_type"},
    )
    
    relationships_vdb = QdrantStorage(
        namespace="relationships",
        global_config=global_config,
        embedding_func=embedding_func,
        meta_fields={"src_id", "tgt_id", "rel_type"},
    )
    
    chunk_entity_relation_graph = NetworkXStorage(
        namespace="chunk_entity_relation_graph",
        global_config=global_config,
        embedding_func=embedding_func,
    )
    
    # Create MiniRAG instance
    minirag = MiniRAG(
        working_dir=working_dir,
        embedding_func=embedding_func,
        full_docs=full_docs,
        text_chunks=text_chunks,
        doc_status=doc_status,
        chunks_vdb=chunks_vdb,
        entities_vdb=entities_vdb,
        entity_name_vdb=entity_name_vdb,
        relationships_vdb=relationships_vdb,
        chunk_entity_relation_graph=chunk_entity_relation_graph,
        global_config=global_config,
    )
    
    return minirag


async def index_documents(minirag: MiniRAG, documents: List[str]) -> List[str]:
    """
    Index documents into MiniRAG
    
    Args:
        minirag: MiniRAG instance
        documents: List of document strings to index
        
    Returns:
        List of document IDs that were indexed
    """
    return await minirag.index(documents)


async def query_minirag(
    minirag: MiniRAG,
    query: str,
    param: Optional[QueryParam] = None,
) -> Dict[str, Any]:
    """
    Query MiniRAG for information
    
    Args:
        minirag: MiniRAG instance
        query: Query string
        param: Query parameters (optional)
        
    Returns:
        Query result
    """
    return await minirag.query(query, param)


def main():
    """Main entry point for the MiniRAG CLI"""
    parser = argparse.ArgumentParser(description="MiniRAG: Minimalist Retrieval-Augmented Generation")
    
    # Set up logging
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    set_logger(os.path.join(log_dir, "minirag.log"))
    
    # Parse arguments
    parser.add_argument("--working-dir", type=str, default="./minirag_data",
                        help="Working directory for storage")
    parser.add_argument("--embedding-model", type=str, default="openai",
                        choices=["openai", "sentence-transformers"],
                        help="Embedding model to use")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--file", type=str, required=True,
                             help="File containing documents to index (one per line)")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query MiniRAG")
    query_parser.add_argument("--query", type=str, required=True,
                             help="Query string")
    query_parser.add_argument("--top-k", type=int, default=60,
                             help="Number of top-k items to retrieve")
    
    args = parser.parse_args()
    
    # Create working directory
    os.makedirs(args.working_dir, exist_ok=True)
    
    # Run command
    if args.command == "index":
        # Read documents from file
        with open(args.file, "r") as f:
            documents = [line.strip() for line in f if line.strip()]
            
        # Index documents
        async def run_index():
            minirag = await create_minirag(args.working_dir, args.embedding_model)
            doc_ids = await index_documents(minirag, documents)
            logger.info(f"Indexed {len(doc_ids)} documents")
            
        asyncio.run(run_index())
        
    elif args.command == "query":
        # Query MiniRAG
        async def run_query():
            minirag = await create_minirag(args.working_dir, args.embedding_model)
            param = QueryParam(top_k=args.top_k)
            result = await query_minirag(minirag, args.query, param)
            
            # Print result
            print(f"Query: {args.query}")
            print("\nRelevant Chunks:")
            for i, chunk in enumerate(result["chunks"]):
                print(f"{i+1}. {chunk['content'][:100]}... (distance: {chunk['distance']:.4f})")
                
            print("\nRelevant Entities:")
            for i, entity in enumerate(result["entities"]):
                print(f"{i+1}. {entity['name']} ({entity['type']}) - {entity['content']}")
                
            print("\nRelationships:")
            for i, rel in enumerate(result["relationships"]):
                print(f"{i+1}. {rel['source']} -> {rel['target']}: {rel['type']}")
                
        asyncio.run(run_query())
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 