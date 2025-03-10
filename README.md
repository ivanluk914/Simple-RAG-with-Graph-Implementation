# Simple-RAG-with-Graph-Implementation
A minimalist Retrieval-Augmented Generation (RAG) system that uses JSON key-value storage, NetworkX for knowledge graphs, and Qdrant for vector storage.

## Features

- Simple and modular architecture
- Default storage implementations:
  - JSON key-value storage for documents and chunks
  - NetworkX for knowledge graph storage
  - Qdrant for vector database
- Entity extraction and relationship building
- Flexible embedding options (OpenAI or Sentence Transformers)
- Command-line interface for indexing and querying

## Simple-RAG
```
simplerag/
├── main.py                      # Main entry point
│
├── core/                        # Core components
│   ├── init.py
│   ├── base.py                  # Core abstract base classes
│   ├── prompt.py                # Prompt templates
│   ├── exceptions.py            # Exception handling
│   └── utils.py                 # Utility functions
│
├── storage/                     # Storage initialization and management
│   ├── init.py
│   ├── initialize.py            # Storage setup and initialization
│   ├── chunking.py              # Implemented by Vincent
│   ├── extract.py               # Entity extraction functionality
│   ├── indexing.py              # Vector indexing and embedding
│   └── implementations/         # Storage implementations
│       ├── init.py
│       ├── kv_storage.py        # JSON key-value storage
│       ├── graph_storage.py     # NetworkX graph storage
│       └── qdrant_storage.py    # Qdrant vector database
│
└── query/                       # Query operations
    ├── init.py
    ├── operates.py              # Core query operations
    └── simplerag.py             # SimpleRAG-specific query functionality
```
