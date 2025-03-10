class MiniRAGException(Exception):
    """Base exception for MiniRAG"""
    pass


class StorageException(MiniRAGException):
    """Exception raised for errors in the storage operations"""
    pass


class EmbeddingException(MiniRAGException):
    """Exception raised for errors in the embedding operations"""
    pass


class ChunkingException(MiniRAGException):
    """Exception raised for errors in the chunking operations"""
    pass


class QueryException(MiniRAGException):
    """Exception raised for errors in the query operations"""
    pass


class LLMException(MiniRAGException):
    """Exception raised for errors in the LLM operations"""
    pass


class ConfigException(MiniRAGException):
    """Exception raised for errors in the configuration"""
    pass


class EntityExtractionException(MiniRAGException):
    """Exception raised for errors in the entity extraction operations"""
    pass 