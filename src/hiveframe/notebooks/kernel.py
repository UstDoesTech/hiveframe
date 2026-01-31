"""
Notebook Kernel - Multi-language Execution Engine
=================================================

Execution engine for notebook cells with multi-language support.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
import sys
from io import StringIO


class KernelLanguage(Enum):
    """Supported kernel languages."""

    PYTHON = "python"
    SQL = "sql"
    R = "r"
    SCALA = "scala"


class CellStatus(Enum):
    """Cell execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class CellOutput:
    """Output from cell execution."""

    output_type: str  # "stream", "execute_result", "error"
    data: Any
    execution_count: Optional[int] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CellExecution:
    """Record of a cell execution."""

    cell_id: str
    code: str
    language: KernelLanguage
    status: CellStatus
    outputs: List[CellOutput] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_count: int = 0


class NotebookKernel:
    """
    Notebook kernel for executing code cells.

    Supports multiple languages with isolated execution contexts,
    similar to how bee colonies maintain separate task groups.

    Example:
        kernel = NotebookKernel(language=KernelLanguage.PYTHON)

        # Execute Python code
        result = kernel.execute(\"\"\"
        x = 10
        y = 20
        x + y
        \"\"\")

        print(f"Output: {result.outputs}")
    """

    def __init__(self, language: KernelLanguage = KernelLanguage.PYTHON, timeout: int = 300):
        """
        Initialize notebook kernel.

        Args:
            language: Kernel language
            timeout: Execution timeout in seconds
        """
        self.language = language
        self.timeout = timeout
        self.execution_count = 0

        # Execution context (namespace for Python)
        self.context: Dict[str, Any] = {}

        # Initialize language-specific context
        if language == KernelLanguage.PYTHON:
            self.context = {"__builtins__": __builtins__}

    def execute(self, code: str, cell_id: Optional[str] = None) -> CellExecution:
        """
        Execute code in the kernel.

        Args:
            code: Code to execute
            cell_id: Optional cell identifier

        Returns:
            CellExecution result
        """
        cell_id = cell_id or f"cell_{self.execution_count}"
        self.execution_count += 1

        execution = CellExecution(
            cell_id=cell_id,
            code=code,
            language=self.language,
            status=CellStatus.RUNNING,
            execution_count=self.execution_count,
            start_time=datetime.now(),
        )

        try:
            if self.language == KernelLanguage.PYTHON:
                output = self._execute_python(code)
            elif self.language == KernelLanguage.SQL:
                output = self._execute_sql(code)
            else:
                raise NotImplementedError(f"Language {self.language.value} not yet implemented")

            execution.status = CellStatus.SUCCESS
            execution.outputs.append(output)

        except Exception as e:
            execution.status = CellStatus.ERROR
            error_output = CellOutput(
                output_type="error",
                data={"ename": type(e).__name__, "evalue": str(e), "traceback": [str(e)]},
            )
            execution.outputs.append(error_output)

        finally:
            execution.end_time = datetime.now()

        return execution

    def _execute_python(self, code: str) -> CellOutput:
        """
        Execute Python code.

        WARNING: This is a simplified implementation for Phase 3.
        Production use should implement proper sandboxing and security measures.
        Consider using RestrictedPython or similar libraries for production.
        """
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()

        try:
            # Execute code
            # NOTE: eval/exec on user code is a security risk
            # This is acceptable for Phase 3 demo but needs sandboxing for production
            result = (
                eval(code, self.context) if self._is_expression(code) else exec(code, self.context)
            )

            # Get captured output
            output_text = captured_output.getvalue()

            # Determine output type
            if result is not None:
                output = CellOutput(
                    output_type="execute_result",
                    data={"text/plain": str(result)},
                    execution_count=self.execution_count,
                )
            elif output_text:
                output = CellOutput(
                    output_type="stream",
                    data={"name": "stdout", "text": output_text},
                    execution_count=self.execution_count,
                )
            else:
                output = CellOutput(
                    output_type="execute_result",
                    data={"text/plain": ""},
                    execution_count=self.execution_count,
                )

            return output

        finally:
            sys.stdout = old_stdout

    def _execute_sql(self, code: str) -> CellOutput:
        """Execute SQL code (stub implementation)."""
        # In a real implementation, this would connect to a SQL engine
        return CellOutput(
            output_type="execute_result",
            data={"text/plain": "SQL execution not yet implemented"},
            execution_count=self.execution_count,
        )

    def _is_expression(self, code: str) -> bool:
        """Check if code is a single expression."""
        try:
            compile(code, "<string>", "eval")
            return True
        except SyntaxError:
            return False

    def reset(self) -> None:
        """Reset kernel state."""
        self.execution_count = 0
        if self.language == KernelLanguage.PYTHON:
            self.context = {"__builtins__": __builtins__}
        else:
            self.context = {}

    def get_context(self) -> Dict[str, Any]:
        """Get current execution context."""
        return self.context.copy()


class NotebookSession:
    """
    Manages a notebook session with multiple kernels.

    Similar to how a bee colony coordinates multiple task groups,
    a session manages multiple kernel instances.

    Example:
        session = NotebookSession()

        # Execute Python cell
        result = session.execute_cell(
            \"\"\"
            import hiveframe
            print("HiveFrame initialized")
            \"\"\",
            language=KernelLanguage.PYTHON
        )

        # Execute SQL cell
        result = session.execute_cell(
            "SELECT * FROM users LIMIT 10",
            language=KernelLanguage.SQL
        )
    """

    def __init__(self):
        """Initialize notebook session."""
        self.kernels: Dict[KernelLanguage, NotebookKernel] = {}
        self.execution_history: List[CellExecution] = []

    def get_kernel(self, language: KernelLanguage) -> NotebookKernel:
        """Get or create kernel for language."""
        if language not in self.kernels:
            self.kernels[language] = NotebookKernel(language=language)
        return self.kernels[language]

    def execute_cell(
        self,
        code: str,
        language: KernelLanguage = KernelLanguage.PYTHON,
        cell_id: Optional[str] = None,
    ) -> CellExecution:
        """
        Execute a notebook cell.

        Args:
            code: Cell code
            language: Programming language
            cell_id: Optional cell identifier

        Returns:
            CellExecution result
        """
        kernel = self.get_kernel(language)
        execution = kernel.execute(code, cell_id)
        self.execution_history.append(execution)
        return execution

    def reset_kernel(self, language: KernelLanguage) -> None:
        """Reset a specific kernel."""
        if language in self.kernels:
            self.kernels[language].reset()

    def reset_all_kernels(self) -> None:
        """Reset all kernels."""
        for kernel in self.kernels.values():
            kernel.reset()

    def get_execution_history(
        self, language: Optional[KernelLanguage] = None
    ) -> List[CellExecution]:
        """
        Get execution history, optionally filtered by language.

        Args:
            language: Optional language filter

        Returns:
            List of cell executions
        """
        if language:
            return [e for e in self.execution_history if e.language == language]
        return self.execution_history
