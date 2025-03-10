from minirag.core.utils import (
    encode_string_by_tiktoken,
    decode_tokens_by_tiktoken,
    logger,
)


def chunking_by_token_size(
    content: str,
    overlap_token_size: int = 100,
    max_token_size: int = 1200,
    tiktoken_model: str = "gpt-4o",
):
    """Split content into chunks by token size
    
    Args:
        content: The text content to chunk
        overlap_token_size: Number of tokens to overlap between chunks
        max_token_size: Maximum number of tokens per chunk
        tiktoken_model: The tiktoken model to use for tokenization
        
    Returns:
        List of dictionaries with chunk information
    """
    tokens = encode_string_by_tiktoken(content, tiktoken_model)
    
    if len(tokens) <= max_token_size:
        return [{"content": content, "tokens": len(tokens), "chunk_order_index": 0}]
    
    chunks = []
    for i in range(0, len(tokens), max_token_size - overlap_token_size):
        chunk_tokens = tokens[i:i + max_token_size]
        chunk_text = decode_tokens_by_tiktoken(chunk_tokens, tiktoken_model)
        chunks.append({
            "content": chunk_text,
            "tokens": len(chunk_tokens),
            "chunk_order_index": len(chunks)
        })
    
    logger.info(f"Split content into {len(chunks)} chunks")
    return chunks 