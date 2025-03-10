import os
from dataclasses import dataclass
from typing import Union, Dict

from minirag.core.utils import (
    logger,
    load_json,
    write_json,
)

from minirag.core.base import (
    DocStatus,
    DocProcessingStatus,
    DocStatusStorage,
)


@dataclass
class JsonDocStatusStorage(DocStatusStorage):
    """JSON implementation of document status storage"""

    def __post_init__(self):
        working_dir = self.global_config["working_dir"]
        self._file_name = os.path.join(working_dir, f"kv_store_{self.namespace}.json")
        self._data = load_json(self._file_name) or {}
        logger.info(f"Loaded document status storage with {len(self._data)} records")

    async def filter_keys(self, data: list[str]) -> set[str]:
        """Return keys that should be processed (not in storage or not successfully processed)"""
        return set(
            [
                k
                for k in data
                if k not in self._data or self._data[k]["status"] != DocStatus.PROCESSED
            ]
        )

    async def get_status_counts(self) -> Dict[str, int]:
        """Get counts of documents in each status"""
        counts = {status: 0 for status in DocStatus}
        for doc in self._data.values():
            counts[doc["status"]] += 1
        return counts

    async def get_failed_docs(self) -> Dict[str, DocProcessingStatus]:
        """Get all failed documents"""
        return {k: v for k, v in self._data.items() if v["status"] == DocStatus.FAILED}

    async def get_pending_docs(self) -> Dict[str, DocProcessingStatus]:
        """Get all pending documents"""
        return {k: v for k, v in self._data.items() if v["status"] == DocStatus.PENDING}

    async def index_done_callback(self):
        """Save data to file after indexing"""
        write_json(self._data, self._file_name)

    async def upsert(self, data: dict[str, dict]):
        """Update or insert document status

        Args:
            data: Dictionary of document IDs and their status data
        """
        self._data.update(data)
        await self.index_done_callback()
        return data

    async def get_by_id(self, id: str):
        return self._data.get(id)

    async def get_by_ids(self, ids: list[str], fields=None):
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

    async def all_keys(self) -> list[str]:
        return list(self._data.keys())

    async def drop(self):
        self._data = {}
        await self.index_done_callback()

    async def delete(self, doc_ids: list[str]):
        """Delete document status by IDs"""
        for doc_id in doc_ids:
            self._data.pop(doc_id, None)
        await self.index_done_callback() 