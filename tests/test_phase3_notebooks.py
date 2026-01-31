"""
Tests for Phase 3 HiveFrame Notebooks
"""

import pytest
import tempfile
import os

from hiveframe.notebooks import (
    NotebookKernel,
    NotebookSession,
    CollaborationManager,
    NotebookFormat,
    GPUCell,
)
from hiveframe.notebooks.kernel import KernelLanguage, CellStatus
from hiveframe.notebooks.collaboration import OperationType
from hiveframe.notebooks.format import CellType


class TestNotebookKernel:
    """Test notebook kernel functionality."""
    
    def test_initialization(self):
        """Test kernel initialization."""
        kernel = NotebookKernel(language=KernelLanguage.PYTHON)
        assert kernel.language == KernelLanguage.PYTHON
        assert kernel.execution_count == 0
    
    def test_execute_simple_expression(self):
        """Test executing a simple expression."""
        kernel = NotebookKernel()
        
        execution = kernel.execute("2 + 2")
        assert execution.status == CellStatus.SUCCESS
        assert len(execution.outputs) == 1
        assert execution.outputs[0].output_type == "execute_result"
    
    def test_execute_print_statement(self):
        """Test executing print statement."""
        kernel = NotebookKernel()
        
        execution = kernel.execute("print('Hello, HiveFrame!')")
        assert execution.status == CellStatus.SUCCESS
        assert len(execution.outputs) == 1
        assert "Hello, HiveFrame!" in str(execution.outputs[0].data)
    
    def test_execute_with_variables(self):
        """Test executing code with variables."""
        kernel = NotebookKernel()
        
        # Set variable
        kernel.execute("x = 10")
        
        # Use variable
        execution = kernel.execute("x * 2")
        assert execution.status == CellStatus.SUCCESS
        # Variable should persist in context
        assert kernel.context.get("x") == 10
    
    def test_execute_error(self):
        """Test handling execution errors."""
        kernel = NotebookKernel()
        
        execution = kernel.execute("1 / 0")
        assert execution.status == CellStatus.ERROR
        assert len(execution.outputs) == 1
        assert execution.outputs[0].output_type == "error"
    
    def test_execution_count(self):
        """Test execution count increments."""
        kernel = NotebookKernel()
        
        kernel.execute("x = 1")
        assert kernel.execution_count == 1
        
        kernel.execute("y = 2")
        assert kernel.execution_count == 2
    
    def test_reset_kernel(self):
        """Test resetting kernel state."""
        kernel = NotebookKernel()
        
        kernel.execute("x = 100")
        assert "x" in kernel.context
        
        kernel.reset()
        assert kernel.execution_count == 0
        assert "x" not in kernel.context


class TestNotebookSession:
    """Test notebook session functionality."""
    
    def test_initialization(self):
        """Test session initialization."""
        session = NotebookSession()
        assert len(session.kernels) == 0
        assert len(session.execution_history) == 0
    
    def test_execute_python_cell(self):
        """Test executing a Python cell."""
        session = NotebookSession()
        
        execution = session.execute_cell(
            "x = 42",
            language=KernelLanguage.PYTHON
        )
        
        assert execution.status == CellStatus.SUCCESS
        assert execution.language == KernelLanguage.PYTHON
    
    def test_multiple_languages(self):
        """Test executing cells in different languages."""
        session = NotebookSession()
        
        # Python cell
        session.execute_cell("x = 1", language=KernelLanguage.PYTHON)
        # SQL cell (will be stub)
        session.execute_cell("SELECT * FROM users", language=KernelLanguage.SQL)
        
        assert len(session.kernels) == 2
        assert KernelLanguage.PYTHON in session.kernels
        assert KernelLanguage.SQL in session.kernels
    
    def test_execution_history(self):
        """Test execution history tracking."""
        session = NotebookSession()
        
        session.execute_cell("x = 1")
        session.execute_cell("y = 2")
        session.execute_cell("z = 3")
        
        history = session.get_execution_history()
        assert len(history) == 3
        
        # Filter by language
        python_history = session.get_execution_history(language=KernelLanguage.PYTHON)
        assert len(python_history) == 3
    
    def test_reset_kernel(self):
        """Test resetting a specific kernel."""
        session = NotebookSession()
        
        session.execute_cell("x = 100", language=KernelLanguage.PYTHON)
        session.reset_kernel(KernelLanguage.PYTHON)
        
        kernel = session.get_kernel(KernelLanguage.PYTHON)
        assert kernel.execution_count == 0
    
    def test_reset_all_kernels(self):
        """Test resetting all kernels."""
        session = NotebookSession()
        
        session.execute_cell("x = 1", language=KernelLanguage.PYTHON)
        session.execute_cell("SELECT 1", language=KernelLanguage.SQL)
        
        session.reset_all_kernels()
        
        for kernel in session.kernels.values():
            assert kernel.execution_count == 0


class TestCollaborationManager:
    """Test collaborative editing."""
    
    def test_create_session(self):
        """Test creating a collaborative session."""
        collab = CollaborationManager()
        
        session = collab.create_session("notebook_123")
        assert session.notebook_id == "notebook_123"
        assert session.session_id is not None
    
    def test_add_user(self):
        """Test adding a user to a session."""
        collab = CollaborationManager()
        session = collab.create_session("notebook_123")
        
        user = collab.add_user(
            session.session_id,
            username="alice",
            email="alice@example.com"
        )
        
        assert user.username == "alice"
        assert user.email == "alice@example.com"
        assert user.color is not None  # Should have assigned color
    
    def test_remove_user(self):
        """Test removing a user from a session."""
        collab = CollaborationManager()
        session = collab.create_session("notebook_123")
        user = collab.add_user(session.session_id, "alice", "alice@example.com")
        
        result = collab.remove_user(session.session_id, user.user_id)
        assert result is True
        
        # User should be removed
        active_users = collab.get_active_users(session.session_id)
        assert len(active_users) == 0
    
    def test_broadcast_operation(self):
        """Test broadcasting an operation."""
        collab = CollaborationManager()
        session = collab.create_session("notebook_123")
        user = collab.add_user(session.session_id, "alice", "alice@example.com")
        
        operation = collab.broadcast_operation(
            session.session_id,
            user.user_id,
            cell_id="cell_1",
            operation_type=OperationType.INSERT,
            position=0,
            data="print('Hello')"
        )
        
        assert operation.operation_type == OperationType.INSERT
        assert operation.user_id == user.user_id
        assert operation.cell_id == "cell_1"
    
    def test_get_operations_since(self):
        """Test getting operations since a timestamp."""
        collab = CollaborationManager()
        session = collab.create_session("notebook_123")
        user = collab.add_user(session.session_id, "alice", "alice@example.com")
        
        from datetime import datetime
        start_time = datetime.now()
        
        # Add some operations
        collab.broadcast_operation(
            session.session_id, user.user_id, "cell_1",
            OperationType.INSERT, 0, "code1"
        )
        collab.broadcast_operation(
            session.session_id, user.user_id, "cell_1",
            OperationType.INSERT, 5, "code2"
        )
        
        operations = collab.get_operations_since(session.session_id, start_time)
        assert len(operations) == 2
    
    def test_get_active_users(self):
        """Test getting active users."""
        collab = CollaborationManager()
        session = collab.create_session("notebook_123")
        
        user1 = collab.add_user(session.session_id, "alice", "alice@example.com")
        user2 = collab.add_user(session.session_id, "bob", "bob@example.com")
        
        active_users = collab.get_active_users(session.session_id)
        assert len(active_users) == 2
    
    def test_close_session(self):
        """Test closing a session."""
        collab = CollaborationManager()
        session = collab.create_session("notebook_123")
        
        result = collab.close_session(session.session_id)
        assert result is True
        
        # Session should be gone
        closed_session = collab.get_session(session.session_id)
        assert closed_session is None


class TestNotebookFormat:
    """Test notebook file format."""
    
    def test_create_code_cell(self):
        """Test creating a code cell."""
        fmt = NotebookFormat()
        
        cell = fmt.create_code_cell("print('Hello')", execution_count=1)
        assert cell.cell_type == CellType.CODE
        assert "print('Hello')" in "".join(cell.source)
        assert cell.execution_count == 1
    
    def test_create_markdown_cell(self):
        """Test creating a markdown cell."""
        fmt = NotebookFormat()
        
        cell = fmt.create_markdown_cell("# Title\nSome text")
        assert cell.cell_type == CellType.MARKDOWN
        assert "# Title" in "".join(cell.source)
    
    def test_write_read_notebook(self):
        """Test writing and reading a notebook."""
        fmt = NotebookFormat()
        
        # Create a notebook
        from hiveframe.notebooks.format import Notebook, NotebookCell, CellType
        notebook = Notebook()
        notebook.cells.append(NotebookCell(
            cell_type=CellType.CODE,
            source=["print('Hello, HiveFrame!')"]
        ))
        notebook.cells.append(NotebookCell(
            cell_type=CellType.MARKDOWN,
            source=["# Test Notebook"]
        ))
        
        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ipynb', delete=False) as f:
            temp_path = f.name
        
        try:
            fmt.write_notebook(notebook, temp_path)
            
            # Read back
            loaded = fmt.read_notebook(temp_path)
            
            assert len(loaded.cells) == 2
            assert loaded.cells[0].cell_type == CellType.CODE
            assert loaded.cells[1].cell_type == CellType.MARKDOWN
        finally:
            os.unlink(temp_path)
    
    def test_add_remove_cell(self):
        """Test adding and removing cells."""
        fmt = NotebookFormat()
        from hiveframe.notebooks.format import Notebook
        
        notebook = Notebook()
        cell1 = fmt.create_code_cell("x = 1")
        cell2 = fmt.create_code_cell("y = 2")
        
        # Add cells
        fmt.add_cell(notebook, cell1)
        fmt.add_cell(notebook, cell2)
        assert len(notebook.cells) == 2
        
        # Remove cell
        result = fmt.remove_cell(notebook, 0)
        assert result is True
        assert len(notebook.cells) == 1
    
    def test_move_cell(self):
        """Test moving cells."""
        fmt = NotebookFormat()
        from hiveframe.notebooks.format import Notebook
        
        notebook = Notebook()
        cell1 = fmt.create_code_cell("cell 1")
        cell2 = fmt.create_code_cell("cell 2")
        cell3 = fmt.create_code_cell("cell 3")
        
        fmt.add_cell(notebook, cell1)
        fmt.add_cell(notebook, cell2)
        fmt.add_cell(notebook, cell3)
        
        # Move cell 0 to position 2
        result = fmt.move_cell(notebook, 0, 2)
        assert result is True
        assert "cell 2" in "".join(notebook.cells[0].source)


class TestGPUCell:
    """Test GPU-accelerated cells."""
    
    def test_initialization(self):
        """Test GPU cell initialization."""
        gpu_cell = GPUCell(auto_detect=True)
        assert len(gpu_cell.devices) > 0  # At least CPU fallback
    
    def test_list_devices(self):
        """Test listing GPU devices."""
        gpu_cell = GPUCell(auto_detect=True)
        devices = gpu_cell.list_devices()
        
        assert len(devices) > 0
        assert devices[0].device_id >= 0
    
    def test_select_device(self):
        """Test device selection."""
        gpu_cell = GPUCell(auto_detect=True)
        device = gpu_cell.select_device()
        
        assert device is not None
        assert device.device_id >= 0
    
    def test_execute_code(self):
        """Test executing GPU code."""
        gpu_cell = GPUCell(auto_detect=True)
        
        task = gpu_cell.execute("""
result = 42
""")
        
        assert task.status == "success"
        # Result should be set in context
    
    def test_execute_error(self):
        """Test handling execution errors."""
        gpu_cell = GPUCell(auto_detect=True)
        
        task = gpu_cell.execute("undefined_variable")
        
        assert task.status == "error"
        assert task.error is not None
    
    def test_get_task(self):
        """Test retrieving a task."""
        gpu_cell = GPUCell(auto_detect=True)
        
        task = gpu_cell.execute("x = 1", task_id="task_123")
        
        retrieved = gpu_cell.get_task("task_123")
        assert retrieved is not None
        assert retrieved.task_id == "task_123"
    
    def test_list_tasks(self):
        """Test listing all tasks."""
        gpu_cell = GPUCell(auto_detect=True)
        
        gpu_cell.execute("x = 1")
        gpu_cell.execute("y = 2")
        
        tasks = gpu_cell.list_tasks()
        assert len(tasks) == 2
