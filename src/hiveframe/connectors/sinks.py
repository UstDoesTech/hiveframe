"""
HiveFrame Data Sinks
====================
Sink implementations for writing data to various destinations.
"""

import csv
import json
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TextIO, TypeVar, Union

from ..utils import ManagedResource

T = TypeVar("T")


# ============================================================================
# Base Sink Interface
# ============================================================================


class DataSink(ManagedResource, Generic[T]):
    """
    Abstract base for all data sinks.

    Provides a uniform interface for writing data to various destinations.
    Extends ManagedResource for lifecycle management.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self._records_written = 0

    @abstractmethod
    def write(self, record: T) -> None:
        """Write a single record."""
        pass

    def write_batch(self, records: List[T]) -> int:
        """Write multiple records. Returns count written."""
        written = 0
        for record in records:
            try:
                self.write(record)
                written += 1
            except Exception:
                self._errors += 1
        return written

    def get_stats(self) -> Dict[str, Any]:
        """Get sink statistics (alias for get_metrics)."""
        stats = super().get_stats()
        stats["records_written"] = self._records_written
        return stats

    # Alias for backward compatibility
    def get_metrics(self) -> Dict[str, Any]:
        """Get sink metrics."""
        return self.get_stats()


# ============================================================================
# File Sinks
# ============================================================================


class JSONLSink(DataSink[Dict[str, Any]]):
    """JSON Lines output sink."""

    def __init__(self, path: Union[str, Path], encoding: str = "utf-8", append: bool = False):
        super().__init__(f"jsonl:{path}")
        self.path = Path(path)
        self.encoding = encoding
        self.append = append
        self._file: Optional[TextIO] = None

    def _do_open(self) -> None:
        mode = "a" if self.append else "w"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, mode, encoding=self.encoding)  # type: ignore[assignment]

    def _do_close(self) -> None:
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None

    def write(self, record: Dict[str, Any]) -> None:
        if not self._is_open:
            raise RuntimeError("Sink not open")

        assert self._file is not None, "File must be open"
        self._file.write(json.dumps(record) + "\n")
        self._records_written += 1


class CSVSink(DataSink[Dict[str, str]]):
    """CSV output sink."""

    def __init__(
        self,
        path: Union[str, Path],
        columns: Optional[List[str]] = None,
        delimiter: str = ",",
        write_header: bool = True,
        encoding: str = "utf-8",
        append: bool = False,
    ):
        super().__init__(f"csv:{path}")
        self.path = Path(path)
        self.columns = columns
        self.delimiter = delimiter
        self.write_header = write_header
        self.encoding = encoding
        self.append = append
        self._file: Optional[TextIO] = None
        self._writer: Optional[Any] = None  # csv.writer type
        self._header_written = False

    def _do_open(self) -> None:
        mode = "a" if self.append else "w"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, mode, encoding=self.encoding, newline="")  # type: ignore[assignment]
        assert self._file is not None
        self._writer = csv.writer(self._file, delimiter=self.delimiter)

        # Don't write header if appending to existing file
        if self.append and self.path.stat().st_size > 0:
            self._header_written = True

    def _do_close(self) -> None:
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
        self._writer = None

    def write(self, record: Dict[str, str]) -> None:
        if not self._is_open:
            raise RuntimeError("Sink not open")

        assert self._writer is not None, "Writer must be initialized"

        # Infer columns from first record if not specified
        if self.columns is None:
            self.columns = list(record.keys())

        # Write header if needed
        if self.write_header and not self._header_written:
            self._writer.writerow(self.columns)
            self._header_written = True

        row = [record.get(col, "") for col in self.columns]
        self._writer.writerow(row)
        self._records_written += 1
