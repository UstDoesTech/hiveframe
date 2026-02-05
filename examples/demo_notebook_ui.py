#!/usr/bin/env python3
"""
HiveFrame Notebook UI Demo
===========================

Launch the Jupyter-style notebook interface for HiveFrame.

Features:
- Interactive notebook cells for Python and SQL
- Execute code with Ctrl+Enter
- Save and load notebooks
- Real-time execution with HiveFrame backend

Run: python demo_notebook_ui.py
"""

import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hiveframe.notebooks import NotebookUIServer


def main():
    """Launch the notebook UI server."""
    print("\n" + "=" * 60)
    print("      🐝 HIVEFRAME NOTEBOOK UI 🐝")
    print("=" * 60)
    print("\nJupyter-style notebook interface for HiveFrame")
    print("\nFeatures:")
    print("  • Python and SQL cell execution")
    print("  • Multi-language support in one notebook")
    print("  • Save and load notebooks (.ipynb format)")
    print("  • Keyboard shortcuts (Ctrl+Enter to run)")
    print("\n" + "=" * 60)

    # Create and start server
    server = NotebookUIServer(port=8888, host="localhost")

    print("\n📝 Example notebook cells to try:")
    print("\n  Python:")
    print("    from hiveframe import HiveDataFrame")
    print("    data = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]")
    print("    df = HiveDataFrame(data)")
    print("    df.show()")
    print("\n  SQL:")
    print("    SELECT * FROM users WHERE age > 25")
    print("\n" + "=" * 60 + "\n")

    try:
        server.start(blocking=True)
    except KeyboardInterrupt:
        print("\n\nShutting down server...")
        server.stop()
        print("Server stopped successfully.")


if __name__ == "__main__":
    main()
