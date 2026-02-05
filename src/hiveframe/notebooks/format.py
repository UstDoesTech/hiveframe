"""
Notebook Format - Read/Write Notebook Files
===========================================

Support for reading and writing notebook files in various formats.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class CellType(Enum):
    """Notebook cell types."""

    CODE = "code"
    MARKDOWN = "markdown"
    RAW = "raw"


@dataclass
class NotebookCell:
    """A notebook cell."""

    cell_type: CellType
    source: List[str]  # Lines of source code/text
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_count: Optional[int] = None
    outputs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class NotebookMetadata:
    """Notebook-level metadata."""

    kernel_info: Dict[str, Any] = field(default_factory=dict)
    language_info: Dict[str, Any] = field(default_factory=dict)
    authors: List[str] = field(default_factory=list)
    created: Optional[datetime] = None
    modified: Optional[datetime] = None


@dataclass
class Notebook:
    """A complete notebook document."""

    cells: List[NotebookCell] = field(default_factory=list)
    metadata: NotebookMetadata = field(default_factory=NotebookMetadata)
    nbformat: int = 4
    nbformat_minor: int = 5


class NotebookFormat:
    """
    Notebook format handler.

    Reads and writes notebooks in various formats, with primary support
    for Jupyter notebook format (.ipynb).

    Example:
        fmt = NotebookFormat()

        # Create a notebook
        notebook = Notebook()
        notebook.cells.append(NotebookCell(
            cell_type=CellType.CODE,
            source=["print('Hello, HiveFrame!')"]
        ))

        # Save to file
        fmt.write_notebook(notebook, "example.ipynb")

        # Load from file
        loaded = fmt.read_notebook("example.ipynb")
    """

    def __init__(self):
        """Initialize notebook format handler."""
        pass

    def read_notebook(self, filepath: str) -> Notebook:
        """
        Read a notebook from file.

        Args:
            filepath: Path to notebook file

        Returns:
            Parsed Notebook
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self._parse_notebook(data)

    def write_notebook(self, notebook: Notebook, filepath: str) -> None:
        """
        Write a notebook to file.

        Args:
            notebook: Notebook to write
            filepath: Output file path
        """
        data = self._serialize_notebook(notebook)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def _parse_notebook(self, data: Dict[str, Any]) -> Notebook:
        """Parse notebook from JSON data."""
        cells = []
        for cell_data in data.get("cells", []):
            cell = NotebookCell(
                cell_type=CellType(cell_data.get("cell_type", "code")),
                source=cell_data.get("source", []),
                metadata=cell_data.get("metadata", {}),
                execution_count=cell_data.get("execution_count"),
                outputs=cell_data.get("outputs", []),
            )
            cells.append(cell)

        metadata_data = data.get("metadata", {})
        metadata = NotebookMetadata(
            kernel_info=metadata_data.get("kernelspec", {}),
            language_info=metadata_data.get("language_info", {}),
            authors=metadata_data.get("authors", []),
        )

        notebook = Notebook(
            cells=cells,
            metadata=metadata,
            nbformat=data.get("nbformat", 4),
            nbformat_minor=data.get("nbformat_minor", 5),
        )

        return notebook

    def _serialize_notebook(self, notebook: Notebook) -> Dict[str, Any]:
        """Serialize notebook to JSON-compatible dict."""
        cells_data: List[Dict[str, Any]] = []
        for cell in notebook.cells:
            cell_data: Dict[str, Any] = {
                "cell_type": cell.cell_type.value,
                "source": cell.source,
                "metadata": cell.metadata,
            }

            if cell.cell_type == CellType.CODE:
                cell_data["execution_count"] = cell.execution_count
                cell_data["outputs"] = cell.outputs

            cells_data.append(cell_data)

        data = {
            "nbformat": notebook.nbformat,
            "nbformat_minor": notebook.nbformat_minor,
            "cells": cells_data,
            "metadata": {
                "kernelspec": notebook.metadata.kernel_info,
                "language_info": notebook.metadata.language_info,
                "authors": notebook.metadata.authors,
            },
        }

        return data

    def create_code_cell(
        self,
        source: str,
        execution_count: Optional[int] = None,
        outputs: Optional[List[Dict[str, Any]]] = None,
    ) -> NotebookCell:
        """
        Create a code cell.

        Args:
            source: Cell source code
            execution_count: Execution count
            outputs: Cell outputs

        Returns:
            Created NotebookCell
        """
        return NotebookCell(
            cell_type=CellType.CODE,
            source=source.splitlines(keepends=True),
            execution_count=execution_count,
            outputs=outputs or [],
        )

    def create_markdown_cell(self, source: str) -> NotebookCell:
        """
        Create a markdown cell.

        Args:
            source: Markdown text

        Returns:
            Created NotebookCell
        """
        return NotebookCell(cell_type=CellType.MARKDOWN, source=source.splitlines(keepends=True))

    def add_cell(self, notebook: Notebook, cell: NotebookCell) -> None:
        """Add a cell to a notebook."""
        notebook.cells.append(cell)

    def remove_cell(self, notebook: Notebook, index: int) -> bool:
        """Remove a cell from a notebook by index."""
        if 0 <= index < len(notebook.cells):
            notebook.cells.pop(index)
            return True
        return False

    def move_cell(self, notebook: Notebook, from_index: int, to_index: int) -> bool:
        """Move a cell within a notebook."""
        if 0 <= from_index < len(notebook.cells) and 0 <= to_index < len(notebook.cells):
            cell = notebook.cells.pop(from_index)
            notebook.cells.insert(to_index, cell)
            return True
        return False
