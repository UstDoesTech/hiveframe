"""
HiveFrame Notebooks (Phase 3)
=============================

Interactive data science environment with multi-language support.

This module provides:
- Multi-language support (Python, R, SQL, Scala)
- Real-time collaboration
- Version control integration
- GPU-accelerated cells
- Interactive visualizations

Key Components:
    - NotebookKernel: Execution engine for notebook cells
    - NotebookSession: Manages notebook state and execution context
    - CollaborationManager: Real-time multi-user collaboration
    - NotebookFormat: Read/write notebook files
    - GPUCell: GPU-accelerated computation cells

Example:
    from hiveframe.notebooks import NotebookKernel, NotebookSession

    kernel = NotebookKernel(language='python')
    session = NotebookSession(kernel)

    result = session.execute_cell(\"\"\"
    from hiveframe import HiveDataFrame
    df = HiveDataFrame.from_csv('data.csv')
    df.show()
    \"\"\")
"""

__all__ = [
    "NotebookKernel",
    "NotebookSession",
    "CollaborationManager",
    "NotebookFormat",
    "GPUCell",
    "KernelLanguage",
    "CellStatus",
    "OperationType",
    "CellType",
]

from .collaboration import CollaborationManager, OperationType
from .format import CellType, NotebookFormat
from .gpu_cell import GPUCell
from .kernel import CellStatus, KernelLanguage, NotebookKernel, NotebookSession
