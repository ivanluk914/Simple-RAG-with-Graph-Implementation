import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union, List
import numpy as np
import tiktoken

logger = logging.getLogger("minirag")


def set_logger(log_file: str):
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)


@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def limit_async_func_call(max_size: int, waitting_time: float = 0.0001):
    """Add restriction of maximum async calling times for a async func"""

    def final_decro(func):
        """Not using async.Semaphore to avoid use nest-asyncio"""
        __current_size = 0

        @wraps(func)
        async def wait_func(*args, **kwargs):
            nonlocal __current_size
            while __current_size >= max_size:
                await asyncio.sleep(waitting_time)
            __current_size += 1
            result = await func(*args, **kwargs)
            __current_size -= 1
            return result

        return wait_func

    return final_decro


def load_json(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r") as f:
            return json.load(f)
    return None


def write_json(json_obj, file_name):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        json.dump(json_obj, f, indent=2)


def encode_string_by_tiktoken(content: str, model_name: str = "gpt-4o"):
    """Encode a string to tokens using tiktoken"""
    encoder = tiktoken.encoding_for_model(model_name)
    return encoder.encode(content)


def decode_tokens_by_tiktoken(tokens: list[int], model_name: str = "gpt-4o"):
    """Decode tokens to a string using tiktoken"""
    encoder = tiktoken.encoding_for_model(model_name)
    return encoder.decode(tokens)


def chunking_by_token_size(
    content: str,
    overlap_token_size: int = 100,
    max_token_size: int = 1200,
    tiktoken_model: str = "gpt-4o",
):
    """Split content into chunks by token size"""
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
    
    return chunks 