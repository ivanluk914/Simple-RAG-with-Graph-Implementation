import asyncio
import os
from dataclasses import dataclass

from minirag.core.utils import (
    logger,
    load_json,
    write_json,
)

from minirag.core.base import (
    BaseKVStorage,
)


@dataclass
class JsonKVStorage(BaseKVStorage):
    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        self._lock = asyncio.Lock()
        logger.info(f"Load KV {self.namespace} with {len(self._data)} data")

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def index_done_callback(self):
        write_json(self._data, self._file_name)

    async def get_by_id(self, id):
        return self._data.get(id, None)

    async def get_by_ids(self, ids, fields=None):
        if fields is None:
            return [self._data.get(id, None) for id in ids]
        return [
            (
                {k: v for k, v in self._data[id].items() if k in fields}
                if self._data.get(id, None)
                else None
            )
            for id in ids
        ]

    async def filter_keys(self, data: list[str]) -> set[str]:
        return set([s for s in data if s not in self._data])

    async def upsert(self, data: dict[str, dict]):
        left_data = {k: v for k, v in data.items() if k not in self._data}
        self._data.update(left_data)
        return left_data

    async def drop(self):
        self._data = {}

    async def filter(self, filter_func):
        """Filter key-value pairs based on a filter function

        Args:
            filter_func: The filter function, which takes a value as an argument and returns a boolean value

        Returns:
            Dict: Key-value pairs that meet the condition
        """
        result = {}
        async with self._lock:
            for key, value in self._data.items():
                if filter_func(value):
                    result[key] = value
        return result

    async def delete(self, ids: list[str]):
        """Delete data with specified IDs

        Args:
            ids: List of IDs to delete
        """
        async with self._lock:
            for id in ids:
                if id in self._data:
                    del self._data[id]
            await self.index_done_callback()
            logger.info(f"Successfully deleted {len(ids)} items from {self.namespace}") 