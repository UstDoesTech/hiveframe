"""
Notebook UI Server
==================

Web-based Jupyter-style UI for notebook authoring and execution.
"""

import json
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Optional
from urllib.parse import parse_qs, urlparse

from .format import CellType, Notebook, NotebookCell, NotebookFormat
from .kernel import KernelLanguage, NotebookSession

# HTML template for the notebook UI
NOTEBOOK_UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐝 HiveFrame Notebook</title>
    <style>
        :root {
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --primary: #f4a261;
            --secondary: #e76f51;
            --accent: #2a9d8f;
            --text: #e9ecef;
            --text-muted: #adb5bd;
            --border: #495057;
            --code-bg: #0f1419;
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: var(--bg-color);
            color: var(--text);
            min-height: 100vh;
        }

        .header {
            background: linear-gradient(135deg, var(--card-bg), #0f3460);
            padding: 1rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid var(--border);
        }

        .header h1 {
            font-size: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .header-actions {
            display: flex;
            gap: 0.5rem;
        }

        .btn {
            padding: 0.5rem 1rem;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.2s;
        }

        .btn:hover {
            background: #238276;
        }

        .btn-secondary {
            background: var(--border);
        }

        .btn-secondary:hover {
            background: #6c757d;
        }

        .btn-danger {
            background: var(--secondary);
        }

        .btn-danger:hover {
            background: #d65a3e;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        .toolbar {
            background: var(--card-bg);
            padding: 1rem;
            border-radius: 4px;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .language-selector {
            padding: 0.5rem;
            background: var(--code-bg);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 0.9rem;
        }

        .notebook-title {
            flex: 1;
            padding: 0.5rem;
            background: var(--code-bg);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 4px;
            font-size: 1rem;
        }

        .cells {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .cell {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 4px;
            overflow: hidden;
        }

        .cell.executing {
            border-color: var(--accent);
        }

        .cell.error {
            border-color: var(--secondary);
        }

        .cell-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.5rem 1rem;
            background: rgba(255, 255, 255, 0.05);
            border-bottom: 1px solid var(--border);
        }

        .cell-type {
            font-size: 0.8rem;
            color: var(--text-muted);
            text-transform: uppercase;
        }

        .cell-actions {
            display: flex;
            gap: 0.5rem;
        }

        .cell-btn {
            padding: 0.25rem 0.5rem;
            background: transparent;
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 3px;
            cursor: pointer;
            font-size: 0.8rem;
        }

        .cell-btn:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        .cell-content {
            padding: 0;
        }

        .cell-input {
            width: 100%;
            min-height: 100px;
            padding: 1rem;
            background: var(--code-bg);
            color: var(--text);
            border: none;
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9rem;
            resize: vertical;
            line-height: 1.5;
        }

        .cell-output {
            padding: 1rem;
            background: rgba(0, 0, 0, 0.2);
            border-top: 1px solid var(--border);
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9rem;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 400px;
            overflow-y: auto;
        }

        .cell-output.error {
            color: #ff6b6b;
        }

        .cell-output.empty {
            display: none;
        }

        .execution-count {
            color: var(--text-muted);
            font-size: 0.8rem;
            margin-left: 0.5rem;
        }

        .status-bar {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background: var(--card-bg);
            padding: 0.5rem 2rem;
            border-top: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.9rem;
        }

        .status-indicator {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: var(--accent);
        }

        .status-dot.idle {
            background: var(--text-muted);
        }

        .status-dot.running {
            background: var(--accent);
            animation: pulse 1s infinite;
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .markdown-preview {
            padding: 1rem;
            line-height: 1.6;
        }

        .markdown-preview h1 { font-size: 2rem; margin: 1rem 0; }
        .markdown-preview h2 { font-size: 1.5rem; margin: 0.8rem 0; }
        .markdown-preview h3 { font-size: 1.2rem; margin: 0.6rem 0; }
        .markdown-preview p { margin: 0.5rem 0; }
        .markdown-preview code {
            background: var(--code-bg);
            padding: 0.2rem 0.4rem;
            border-radius: 3px;
            font-family: monospace;
        }
        .markdown-preview pre {
            background: var(--code-bg);
            padding: 1rem;
            border-radius: 4px;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐝 HiveFrame Notebook</h1>
        <div class="header-actions">
            <button class="btn btn-secondary" onclick="newNotebook()">New</button>
            <button class="btn btn-secondary" onclick="loadNotebook()">Load</button>
            <button class="btn" onclick="saveNotebook()">Save</button>
            <button class="btn" onclick="runAll()">Run All</button>
        </div>
    </div>

    <div class="container">
        <div class="toolbar">
            <input type="text" class="notebook-title" id="notebookTitle" 
                   value="Untitled Notebook" placeholder="Notebook title">
            <select class="language-selector" id="defaultLanguage">
                <option value="python">Python</option>
                <option value="sql">SQL</option>
            </select>
            <button class="btn" onclick="addCell()">+ Add Cell</button>
        </div>

        <div class="cells" id="cells">
            <!-- Cells will be added dynamically -->
        </div>
    </div>

    <div class="status-bar">
        <div class="status-indicator">
            <div class="status-dot idle" id="statusDot"></div>
            <span id="statusText">Idle</span>
        </div>
        <div id="cellCount">0 cells</div>
    </div>

    <script>
        let cells = [];
        let nextCellId = 1;
        let notebookSession = null;

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            addCell(); // Start with one cell
            updateCellCount();
        });

        function addCell(index = null, content = '', language = null, outputs = null) {
            const cellId = `cell_${nextCellId++}`;
            const cellLanguage = language || document.getElementById('defaultLanguage').value;
            
            const cell = {
                id: cellId,
                language: cellLanguage,
                content: content,
                outputs: outputs || [],
                executionCount: null
            };

            if (index === null) {
                cells.push(cell);
            } else {
                cells.splice(index, 0, cell);
            }

            renderCells();
            updateCellCount();
            
            // Focus on the new cell
            setTimeout(() => {
                const textarea = document.querySelector(`#${cellId} textarea`);
                if (textarea) textarea.focus();
            }, 100);
        }

        function deleteCell(cellId) {
            const index = cells.findIndex(c => c.id === cellId);
            if (index !== -1) {
                cells.splice(index, 1);
                renderCells();
                updateCellCount();
            }
        }

        function moveCell(cellId, direction) {
            const index = cells.findIndex(c => c.id === cellId);
            if (index === -1) return;

            const newIndex = direction === 'up' ? index - 1 : index + 1;
            if (newIndex < 0 || newIndex >= cells.length) return;

            const cell = cells.splice(index, 1)[0];
            cells.splice(newIndex, 0, cell);
            renderCells();
        }

        function updateCellContent(cellId, content) {
            const cell = cells.find(c => c.id === cellId);
            if (cell) {
                cell.content = content;
            }
        }

        async function runCell(cellId) {
            const cell = cells.find(c => c.id === cellId);
            if (!cell) return;

            const cellElement = document.getElementById(cellId);
            cellElement.classList.add('executing');
            
            setStatus('running', 'Executing...');

            try {
                const response = await fetch('/api/notebook/execute', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        code: cell.content,
                        language: cell.language,
                        cell_id: cellId
                    })
                });

                const result = await response.json();
                
                if (result.success) {
                    cell.outputs = result.data.outputs;
                    cell.executionCount = result.data.execution_count;
                    cellElement.classList.remove('executing');
                    
                    if (result.data.status === 'error') {
                        cellElement.classList.add('error');
                    } else {
                        cellElement.classList.remove('error');
                    }
                } else {
                    cell.outputs = [{
                        output_type: 'error',
                        data: { 'text/plain': result.error }
                    }];
                    cellElement.classList.add('error');
                }

                renderCells();
            } catch (error) {
                cell.outputs = [{
                    output_type: 'error',
                    data: { 'text/plain': `Error: ${error.message}` }
                }];
                cellElement.classList.add('error');
                renderCells();
            } finally {
                cellElement.classList.remove('executing');
                setStatus('idle', 'Idle');
            }
        }

        async function runAll() {
            for (const cell of cells) {
                await runCell(cell.id);
            }
        }

        function renderCells() {
            const container = document.getElementById('cells');
            container.innerHTML = cells.map((cell, index) => `
                <div class="cell" id="${cell.id}">
                    <div class="cell-header">
                        <div>
                            <span class="cell-type">${cell.language}</span>
                            ${cell.executionCount !== null ? 
                                `<span class="execution-count">[${cell.executionCount}]</span>` : ''}
                        </div>
                        <div class="cell-actions">
                            ${index > 0 ? `<button class="cell-btn" onclick="moveCell('${cell.id}', 'up')">↑</button>` : ''}
                            ${index < cells.length - 1 ? `<button class="cell-btn" onclick="moveCell('${cell.id}', 'down')">↓</button>` : ''}
                            <button class="cell-btn" onclick="runCell('${cell.id}')">▶ Run</button>
                            <button class="cell-btn btn-danger" onclick="deleteCell('${cell.id}')">Delete</button>
                        </div>
                    </div>
                    <div class="cell-content">
                        <textarea class="cell-input" 
                                  onchange="updateCellContent('${cell.id}', this.value)"
                                  oninput="updateCellContent('${cell.id}', this.value)"
                                  placeholder="Enter ${cell.language} code...">${cell.content}</textarea>
                        ${renderOutputs(cell)}
                    </div>
                </div>
            `).join('');
        }

        function renderOutputs(cell) {
            if (!cell.outputs || cell.outputs.length === 0) {
                return '';
            }

            return cell.outputs.map(output => {
                const isError = output.output_type === 'error';
                let text = '';

                if (output.data) {
                    if (output.data['text/plain']) {
                        text = output.data['text/plain'];
                    } else if (output.data.text) {
                        text = output.data.text;
                    } else if (output.data.ename && output.data.evalue) {
                        text = `${output.data.ename}: ${output.data.evalue}`;
                    }
                }

                return `<div class="cell-output ${isError ? 'error' : ''}">${escapeHtml(text)}</div>`;
            }).join('');
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        async function saveNotebook() {
            const title = document.getElementById('notebookTitle').value;
            const notebook = {
                title: title,
                cells: cells.map(cell => ({
                    cell_type: 'code',
                    language: cell.language,
                    source: cell.content,
                    outputs: cell.outputs,
                    execution_count: cell.executionCount
                }))
            };

            try {
                const response = await fetch('/api/notebook/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(notebook)
                });

                const result = await response.json();
                if (result.success) {
                    alert(`Notebook saved: ${result.data.filepath}`);
                } else {
                    alert(`Error saving notebook: ${result.error}`);
                }
            } catch (error) {
                alert(`Error saving notebook: ${error.message}`);
            }
        }

        async function loadNotebook() {
            const filename = prompt('Enter notebook filename:', 'notebook.ipynb');
            if (!filename) return;

            try {
                const response = await fetch(`/api/notebook/load?filename=${encodeURIComponent(filename)}`);
                const result = await response.json();

                if (result.success) {
                    cells = [];
                    nextCellId = 1;
                    
                    const notebook = result.data;
                    document.getElementById('notebookTitle').value = notebook.title || 'Untitled Notebook';
                    
                    for (const cell of notebook.cells) {
                        addCell(null, cell.source, cell.language || 'python', cell.outputs);
                    }
                } else {
                    alert(`Error loading notebook: ${result.error}`);
                }
            } catch (error) {
                alert(`Error loading notebook: ${error.message}`);
            }
        }

        function newNotebook() {
            if (confirm('Create a new notebook? Unsaved changes will be lost.')) {
                cells = [];
                nextCellId = 1;
                document.getElementById('notebookTitle').value = 'Untitled Notebook';
                addCell();
                updateCellCount();
            }
        }

        function updateCellCount() {
            document.getElementById('cellCount').textContent = `${cells.length} cell${cells.length !== 1 ? 's' : ''}`;
        }

        function setStatus(status, text) {
            const dot = document.getElementById('statusDot');
            const statusText = document.getElementById('statusText');
            
            dot.className = `status-dot ${status}`;
            statusText.textContent = text;
        }

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl+Enter or Cmd+Enter to run cell
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                const activeElement = document.activeElement;
                if (activeElement.tagName === 'TEXTAREA') {
                    const cellElement = activeElement.closest('.cell');
                    if (cellElement) {
                        runCell(cellElement.id);
                    }
                }
                e.preventDefault();
            }
            
            // Shift+Enter to run cell and move to next
            if (e.shiftKey && e.key === 'Enter') {
                const activeElement = document.activeElement;
                if (activeElement.tagName === 'TEXTAREA') {
                    const cellElement = activeElement.closest('.cell');
                    if (cellElement) {
                        const cellId = cellElement.id;
                        const index = cells.findIndex(c => c.id === cellId);
                        runCell(cellId).then(() => {
                            if (index === cells.length - 1) {
                                addCell();
                            } else {
                                const nextCell = document.getElementById(cells[index + 1].id);
                                if (nextCell) {
                                    nextCell.querySelector('textarea').focus();
                                }
                            }
                        });
                    }
                }
                e.preventDefault();
            }
        });
    </script>
</body>
</html>
"""


class NotebookUIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for notebook UI."""

    def __init__(self, *args, session: Optional[NotebookSession] = None, **kwargs):
        self.session = session or NotebookSession()
        self.notebook_format = NotebookFormat()
        super().__init__(*args, **kwargs)

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/notebook":
            self._serve_ui()
        elif path == "/api/notebook/load":
            self._handle_load_notebook(parsed)
        else:
            self.send_error(404, "Not found")

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/notebook/execute":
            self._handle_execute_cell()
        elif path == "/api/notebook/save":
            self._handle_save_notebook()
        else:
            self.send_error(404, "Not found")

    def _serve_ui(self):
        """Serve the notebook UI HTML."""
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(NOTEBOOK_UI_HTML.encode())

    def _handle_execute_cell(self):
        """Handle cell execution request."""
        try:
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())

            code = data.get("code", "")
            language_str = data.get("language", "python")
            cell_id = data.get("cell_id")

            # Map language string to enum
            language_map = {
                "python": KernelLanguage.PYTHON,
                "sql": KernelLanguage.SQL,
            }
            language = language_map.get(language_str.lower(), KernelLanguage.PYTHON)

            # Execute the cell
            execution = self.session.execute_cell(code, language=language, cell_id=cell_id)

            # Format outputs
            outputs = []
            for output in execution.outputs:
                outputs.append(
                    {
                        "output_type": output.output_type,
                        "data": output.data,
                        "execution_count": output.execution_count,
                    }
                )

            response = {
                "success": True,
                "data": {
                    "status": execution.status.value,
                    "outputs": outputs,
                    "execution_count": execution.execution_count,
                },
            }

        except Exception as e:
            response = {"success": False, "error": str(e)}

        self._send_json_response(response)

    def _handle_save_notebook(self):
        """Handle save notebook request."""
        try:
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)
            data = json.loads(body.decode())

            title = data.get("title", "untitled")
            cells_data = data.get("cells", [])

            # Create notebook
            notebook = Notebook()
            for cell_data in cells_data:
                cell = NotebookCell(
                    cell_type=CellType.CODE,
                    source=[cell_data.get("source", "")],
                    metadata={"language": cell_data.get("language", "python")},
                    execution_count=cell_data.get("execution_count"),
                    outputs=cell_data.get("outputs", []),
                )
                notebook.cells.append(cell)

            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{title.replace(' ', '_')}_{timestamp}.ipynb"
            filepath = f"/tmp/{filename}"

            # Save notebook
            self.notebook_format.write_notebook(notebook, filepath)

            response = {"success": True, "data": {"filepath": filepath, "filename": filename}}

        except Exception as e:
            response = {"success": False, "error": str(e)}

        self._send_json_response(response)

    def _handle_load_notebook(self, parsed):
        """Handle load notebook request."""
        try:
            query = parse_qs(parsed.query)
            filename = query.get("filename", [""])[0]

            if not filename:
                raise ValueError("No filename provided")

            filepath = f"/tmp/{filename}"

            # Load notebook
            notebook = self.notebook_format.read_notebook(filepath)

            # Convert to response format
            cells = []
            for cell in notebook.cells:
                cells.append(
                    {
                        "cell_type": cell.cell_type.value,
                        "source": "".join(cell.source),
                        "language": cell.metadata.get("language", "python"),
                        "outputs": cell.outputs,
                        "execution_count": cell.execution_count,
                    }
                )

            response = {"success": True, "data": {"title": filename.replace(".ipynb", ""), "cells": cells}}

        except Exception as e:
            response = {"success": False, "error": str(e)}

        self._send_json_response(response)

    def _send_json_response(self, data: Dict):
        """Send JSON response."""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        """Override to reduce log noise."""
        pass


class NotebookUIServer:
    """
    Notebook UI Server
    ------------------

    Web server for the Jupyter-style notebook UI.

    Example:
        server = NotebookUIServer(port=8888)
        server.start()
        # Open browser to http://localhost:8888
    """

    def __init__(self, port: int = 8888, host: str = "localhost"):
        """
        Initialize server.

        Args:
            port: Server port
            host: Server host
        """
        self.port = port
        self.host = host
        self.session = NotebookSession()
        self.server = None

    def start(self, blocking: bool = True):
        """
        Start the server.

        Args:
            blocking: If True, blocks until server is stopped
        """

        def handler(*args, **kwargs):
            NotebookUIHandler(*args, session=self.session, **kwargs)

        self.server = HTTPServer((self.host, self.port), handler)

        print(f"🐝 HiveFrame Notebook UI running at http://{self.host}:{self.port}")
        print("Press Ctrl+C to stop")

        if blocking:
            try:
                self.server.serve_forever()
            except KeyboardInterrupt:
                print("\nShutting down...")
                self.stop()
        else:
            import threading

            thread = threading.Thread(target=self.server.serve_forever)
            thread.daemon = True
            thread.start()

    def stop(self):
        """Stop the server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
