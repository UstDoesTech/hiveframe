# HiveFrame Notebook UI

A Jupyter-style web interface for interactive notebook authoring and execution in HiveFrame.

## Features

- **Multi-language Support**: Execute Python and SQL code in the same notebook
- **Interactive Cells**: Add, remove, and reorder cells dynamically
- **Real-time Execution**: Run cells individually or all at once
- **Notebook Management**: Save and load notebooks in standard .ipynb format
- **Integrated SQL Engine**: SQL cells use HiveFrame's SwarmQL engine
- **Keyboard Shortcuts**: 
  - `Ctrl+Enter` / `Cmd+Enter`: Run current cell
  - `Shift+Enter`: Run cell and move to next

## Quick Start

### Starting the Notebook Server

```python
from hiveframe.notebooks import NotebookUIServer

# Create and start the server
server = NotebookUIServer(port=8888)
server.start()
```

Or use the demo script:

```bash
python examples/demo_notebook_ui.py
```

Then open your browser to `http://localhost:8888`

### Python Cells

Python cells have access to a shared `_sql_context` for registering tables:

```python
from hiveframe import HiveDataFrame

# Create data
data = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25}
]

# Create DataFrame
df = HiveDataFrame(data)

# Register for SQL queries
_sql_context.register_table("users", df)

df.show()
```

### SQL Cells

SQL cells execute queries using HiveFrame's SwarmQL engine:

```sql
SELECT name, age 
FROM users 
WHERE age > 25
ORDER BY age DESC
```

## Architecture

The notebook UI consists of:

1. **NotebookUIServer**: HTTP server hosting the web interface
2. **NotebookSession**: Manages kernel state across cells
3. **NotebookKernel**: Executes code in Python or SQL
4. **SwarmQL Integration**: SQL execution via HiveFrame's query engine

## API

### Execute a Cell

```http
POST /api/notebook/execute
Content-Type: application/json

{
  "code": "print('Hello')",
  "language": "python",
  "cell_id": "cell_1"
}
```

### Save Notebook

```http
POST /api/notebook/save
Content-Type: application/json

{
  "title": "My Notebook",
  "cells": [
    {
      "cell_type": "code",
      "language": "python",
      "source": "print('Hello')"
    }
  ]
}
```

### Load Notebook

```http
GET /api/notebook/load?filename=notebook.ipynb
```

## Examples

See `examples/sample_notebook.ipynb` for a complete example notebook.

## Integration with HiveFrame

The notebook interface is fully integrated with HiveFrame:

- **DataFrame API**: Use HiveDataFrame in Python cells
- **SQL Engine**: Query registered tables with SwarmQL
- **Shared Context**: Python and SQL cells share the same SQL context
- **Native Format**: Notebooks use standard Jupyter .ipynb format

## Development

### Running Tests

```bash
pytest tests/test_phase3_notebooks.py -v
```

### Adding Features

The notebook system is modular:

- `ui.py`: Web interface and HTTP handlers
- `kernel.py`: Code execution engines
- `format.py`: Notebook file format handlers
- `collaboration.py`: Multi-user support (future)

## Security Note

**Warning**: The current implementation uses `eval()` and `exec()` for code execution. This is suitable for Phase 3 demonstration but requires proper sandboxing for production use. Consider using RestrictedPython or similar libraries before deploying in multi-user environments.

## Future Enhancements

- Real-time collaboration (already implemented in `collaboration.py`)
- GPU-accelerated cells (already implemented in `gpu_cell.py`)
- Rich output formats (plots, images, HTML)
- Notebook templates and examples
- Cell execution queuing and cancellation
- Markdown cell rendering
- Code completion and syntax highlighting
