"""
Dashboard Server
================

HTTP server for the Colony Dashboard.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import os

from .api import DashboardAPI

# HTML template for the dashboard
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🐝 HiveFrame Colony Dashboard</title>
    <style>
        :root {
            --bg-color: #1a1a2e;
            --card-bg: #16213e;
            --primary: #f4a261;
            --secondary: #e76f51;
            --accent: #2a9d8f;
            --text: #e9ecef;
            --text-muted: #adb5bd;
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
            padding: 1.5rem 2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .header h1 {
            font-size: 1.8rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .header .status {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--accent);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
            padding: 1.5rem;
        }
        
        .card {
            background: var(--card-bg);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(255,255,255,0.05);
        }
        
        .card h2 {
            font-size: 1.1rem;
            margin-bottom: 1rem;
            color: var(--primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1rem;
        }
        
        .metric {
            background: rgba(0,0,0,0.2);
            padding: 1rem;
            border-radius: 8px;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-bottom: 0.25rem;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 600;
        }
        
        .metric-value.temperature {
            color: var(--secondary);
        }
        
        .metric-value.fitness {
            color: var(--accent);
        }
        
        .worker-list {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .worker-item {
            display: flex;
            align-items: center;
            padding: 0.75rem;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .worker-role {
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-right: 0.75rem;
        }
        
        .role-employed { background: var(--primary); color: #000; }
        .role-onlooker { background: var(--accent); color: #000; }
        .role-scout { background: var(--secondary); color: #fff; }
        
        .worker-id {
            flex: 1;
            font-family: monospace;
        }
        
        .worker-status {
            font-size: 0.85rem;
            color: var(--text-muted);
        }
        
        .sql-console {
            grid-column: 1 / -1;
        }
        
        .sql-input {
            width: 100%;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 1rem;
            color: var(--text);
            font-family: monospace;
            font-size: 0.9rem;
            resize: vertical;
            min-height: 100px;
        }
        
        .sql-input:focus {
            outline: none;
            border-color: var(--primary);
        }
        
        .sql-actions {
            margin-top: 1rem;
            display: flex;
            gap: 0.5rem;
        }
        
        .btn {
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            transition: opacity 0.2s;
        }
        
        .btn:hover { opacity: 0.8; }
        
        .btn-primary {
            background: var(--primary);
            color: #000;
        }
        
        .btn-secondary {
            background: rgba(255,255,255,0.1);
            color: var(--text);
        }
        
        .results-table {
            width: 100%;
            margin-top: 1rem;
            border-collapse: collapse;
            font-size: 0.85rem;
        }
        
        .results-table th {
            background: rgba(0,0,0,0.3);
            padding: 0.75rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        
        .results-table td {
            padding: 0.5rem 0.75rem;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        
        .dance-item {
            display: flex;
            align-items: center;
            padding: 0.5rem;
            border-left: 3px solid var(--primary);
            margin-bottom: 0.5rem;
            background: rgba(0,0,0,0.2);
            border-radius: 0 4px 4px 0;
        }
        
        .dance-vigor {
            width: 60px;
            height: 8px;
            background: rgba(255,255,255,0.1);
            border-radius: 4px;
            overflow: hidden;
            margin-left: auto;
        }
        
        .dance-vigor-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--secondary), var(--primary));
            transition: width 0.3s;
        }
        
        .chart-container {
            height: 150px;
            margin-top: 1rem;
            position: relative;
        }
        
        .chart-placeholder {
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🐝 Colony Dashboard</h1>
        <div class="status">
            <div class="status-dot"></div>
            <span id="connection-status">Connected</span>
        </div>
    </div>
    
    <div class="dashboard">
        <div class="card">
            <h2>📊 Colony Metrics</h2>
            <div class="metric-grid">
                <div class="metric">
                    <div class="metric-label">Temperature</div>
                    <div class="metric-value temperature" id="metric-temp">0.50</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Fitness</div>
                    <div class="metric-value fitness" id="metric-fitness">0.00</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Food Sources</div>
                    <div class="metric-value" id="metric-sources">0</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Throughput</div>
                    <div class="metric-value" id="metric-throughput">0/s</div>
                </div>
            </div>
            <div class="chart-container">
                <div class="chart-placeholder">📈 Metrics Chart</div>
            </div>
        </div>
        
        <div class="card">
            <h2>🐝 Workers</h2>
            <div class="metric-grid" style="margin-bottom: 1rem;">
                <div class="metric">
                    <div class="metric-label">Total Workers</div>
                    <div class="metric-value" id="worker-total">0</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Active</div>
                    <div class="metric-value" id="worker-active">0</div>
                </div>
            </div>
            <div class="worker-list" id="worker-list">
                <!-- Workers will be populated here -->
            </div>
        </div>
        
        <div class="card">
            <h2>💃 Dance Floor</h2>
            <div class="metric-grid" style="margin-bottom: 1rem;">
                <div class="metric">
                    <div class="metric-label">Total Dances</div>
                    <div class="metric-value" id="dance-total">0</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Avg Vigor</div>
                    <div class="metric-value" id="dance-vigor">0.00</div>
                </div>
            </div>
            <div id="dance-list">
                <!-- Dances will be populated here -->
            </div>
        </div>
        
        <div class="card">
            <h2>📜 Recent Queries</h2>
            <div id="query-list">
                <!-- Queries will be populated here -->
            </div>
        </div>
        
        <div class="card sql-console">
            <h2>⌨️ SQL Console</h2>
            <textarea class="sql-input" id="sql-input" placeholder="Enter SQL query...
Example: SELECT * FROM users WHERE age > 21"></textarea>
            <div class="sql-actions">
                <button class="btn btn-primary" onclick="executeQuery()">▶ Execute</button>
                <button class="btn btn-secondary" onclick="explainQuery()">📋 Explain</button>
                <button class="btn btn-secondary" onclick="clearResults()">🗑️ Clear</button>
            </div>
            <div id="query-results"></div>
        </div>
    </div>
    
    <script>
        // API base URL
        const API_BASE = '/api';
        
        // Fetch data from API
        async function fetchAPI(endpoint) {
            try {
                const response = await fetch(API_BASE + endpoint);
                return await response.json();
            } catch (e) {
                console.error('API error:', e);
                return { success: false, error: e.message };
            }
        }
        
        // Update metrics
        async function updateMetrics() {
            const result = await fetchAPI('/metrics');
            if (result.success) {
                document.getElementById('metric-temp').textContent = 
                    result.data.temperature.toFixed(2);
                document.getElementById('metric-fitness').textContent = 
                    result.data.averageFitness.toFixed(2);
                document.getElementById('metric-sources').textContent = 
                    result.data.activeFoodSources;
                document.getElementById('metric-throughput').textContent = 
                    result.data.throughput.toFixed(1) + '/s';
            }
        }
        
        // Update workers
        async function updateWorkers() {
            const result = await fetchAPI('/workers');
            if (result.success) {
                document.getElementById('worker-total').textContent = 
                    result.data.totalWorkers;
                document.getElementById('worker-active').textContent = 
                    result.data.statusDistribution.active;
                    
                const list = document.getElementById('worker-list');
                list.innerHTML = result.data.workers.map(w => `
                    <div class="worker-item">
                        <span class="worker-role role-${w.role.toLowerCase()}">${w.role}</span>
                        <span class="worker-id">${w.workerId}</span>
                        <span class="worker-status">${w.status}</span>
                    </div>
                `).join('');
            }
        }
        
        // Update dances
        async function updateDances() {
            const result = await fetchAPI('/dances');
            if (result.success) {
                document.getElementById('dance-total').textContent = 
                    result.data.totalDances;
                document.getElementById('dance-vigor').textContent = 
                    result.data.averageVigor.toFixed(2);
                    
                const list = document.getElementById('dance-list');
                list.innerHTML = result.data.recentDances.map(d => `
                    <div class="dance-item">
                        <span>${d.partitionId}</span>
                        <div class="dance-vigor">
                            <div class="dance-vigor-fill" style="width: ${d.vigor * 100}%"></div>
                        </div>
                    </div>
                `).join('');
            }
        }
        
        // Update queries
        async function updateQueries() {
            const result = await fetchAPI('/queries');
            if (result.success) {
                const list = document.getElementById('query-list');
                list.innerHTML = result.data.queries.map(q => `
                    <div class="worker-item">
                        <span class="worker-role ${q.status === 'COMPLETED' ? 'role-onlooker' : 
                                                    q.status === 'FAILED' ? 'role-scout' : 'role-employed'}">
                            ${q.status}
                        </span>
                        <span class="worker-id">${q.sql}</span>
                        ${q.duration ? `<span class="worker-status">${q.duration.toFixed(2)}s</span>` : ''}
                    </div>
                `).join('');
            }
        }
        
        // Execute SQL query
        async function executeQuery() {
            const sql = document.getElementById('sql-input').value;
            if (!sql.trim()) return;
            
            const response = await fetch(API_BASE + '/queries', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sql })
            });
            
            const result = await response.json();
            const resultsDiv = document.getElementById('query-results');
            
            if (result.success && result.data.rows) {
                const columns = result.data.columns;
                const rows = result.data.rows;
                
                let html = '<table class="results-table"><thead><tr>';
                html += columns.map(c => `<th>${c}</th>`).join('');
                html += '</tr></thead><tbody>';
                html += rows.slice(0, 100).map(row => 
                    '<tr>' + columns.map(c => `<td>${row[c] ?? ''}</td>`).join('') + '</tr>'
                ).join('');
                html += '</tbody></table>';
                
                if (result.data.truncated) {
                    html += `<p style="margin-top: 1rem; color: var(--text-muted);">
                        Showing 100 of ${result.data.totalRows} rows</p>`;
                }
                
                resultsDiv.innerHTML = html;
            } else {
                resultsDiv.innerHTML = `<p style="color: var(--secondary); margin-top: 1rem;">
                    Error: ${result.error || 'Unknown error'}</p>`;
            }
            
            updateQueries();
        }
        
        // Explain query
        async function explainQuery() {
            const sql = document.getElementById('sql-input').value;
            if (!sql.trim()) return;
            
            const response = await fetch(API_BASE + '/queries/explain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sql })
            });
            
            const result = await response.json();
            const resultsDiv = document.getElementById('query-results');
            
            if (result.success) {
                resultsDiv.innerHTML = `<pre style="margin-top: 1rem; background: rgba(0,0,0,0.2); 
                    padding: 1rem; border-radius: 8px; overflow-x: auto;">${result.data.plan}</pre>`;
            } else {
                resultsDiv.innerHTML = `<p style="color: var(--secondary); margin-top: 1rem;">
                    Error: ${result.error}</p>`;
            }
        }
        
        // Clear results
        function clearResults() {
            document.getElementById('query-results').innerHTML = '';
            document.getElementById('sql-input').value = '';
        }
        
        // Initial load and periodic updates
        updateMetrics();
        updateWorkers();
        updateDances();
        updateQueries();
        
        setInterval(updateMetrics, 5000);
        setInterval(updateWorkers, 5000);
        setInterval(updateDances, 5000);
        setInterval(updateQueries, 10000);
    </script>
</body>
</html>"""


@dataclass
class DashboardConfig:
    """
    Dashboard Configuration
    -----------------------
    Configuration options for the Colony Dashboard.
    """

    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    enable_cors: bool = True
    auth_enabled: bool = False
    auth_token: Optional[str] = None


class DashboardRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the dashboard."""

    # Reference to API (set by server)
    api: Optional[DashboardAPI] = None
    config: Optional[DashboardConfig] = None

    def log_message(self, format, *args):
        """Suppress default logging."""
        if self.config and self.config.debug:
            BaseHTTPRequestHandler.log_message(self, format, *args)

    def send_cors_headers(self):
        """Send CORS headers."""
        if self.config and self.config.enable_cors:
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        """Handle OPTIONS requests."""
        self.send_response(200)
        self.send_cors_headers()
        self.end_headers()

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        # Serve dashboard HTML
        if path == "/" or path == "/dashboard":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_cors_headers()
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
            return

        # API endpoints
        if path.startswith("/api/"):
            self.handle_api_get(path[5:])
            return

        # 404
        self.send_error(404, "Not Found")

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path

        if path.startswith("/api/"):
            self.handle_api_post(path[5:])
            return

        self.send_error(404, "Not Found")

    def handle_api_get(self, endpoint: str):
        """Handle API GET requests."""
        if not self.api:
            self.send_error(500, "API not initialized")
            return

        response = None

        if endpoint == "metrics":
            response = self.api.get_metrics()
        elif endpoint == "workers":
            response = self.api.get_workers()
        elif endpoint.startswith("workers/"):
            worker_id = endpoint[8:]
            response = self.api.get_worker(worker_id)
        elif endpoint == "dances":
            response = self.api.get_dances()
        elif endpoint == "queries":
            response = self.api.get_queries()
        elif endpoint.startswith("queries/"):
            query_id = endpoint[8:]
            response = self.api.get_query(query_id)
        elif endpoint == "health":
            response = self.api.get_health()
        elif endpoint == "tables":
            response = self.api.get_tables()
        elif endpoint.startswith("tables/"):
            table_name = endpoint[7:]
            response = self.api.get_table_schema(table_name)
        else:
            self.send_error(404, "Endpoint not found")
            return

        self.send_json_response(response)

    def handle_api_post(self, endpoint: str):
        """Handle API POST requests."""
        if not self.api:
            self.send_error(500, "API not initialized")
            return

        # Read request body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode()

        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        response = None

        if endpoint == "queries":
            sql = data.get("sql", "")
            response = self.api.execute_query(sql)
        elif endpoint == "queries/explain":
            sql = data.get("sql", "")
            response = self.api.explain_query(sql)
        elif endpoint == "config":
            response = self.api.update_config(data)
        elif endpoint == "metrics":
            response = self.api.update_metrics(data)
        else:
            self.send_error(404, "Endpoint not found")
            return

        self.send_json_response(response)

    def send_json_response(self, response):
        """Send JSON API response."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_cors_headers()
        self.end_headers()
        self.wfile.write(response.to_json().encode())


class Dashboard:
    """
    Colony Dashboard
    ----------------

    Web dashboard for monitoring HiveFrame clusters.

    Usage:
        from hiveframe.dashboard import Dashboard

        # Create dashboard
        dashboard = Dashboard(port=8080)

        # Connect to HiveFrame
        dashboard.connect_hive(hive)
        dashboard.connect_sql(sql_context)

        # Start server
        dashboard.start()

        # Access at http://localhost:8080
    """

    def __init__(self, config: Optional[DashboardConfig] = None, port: int = 8080):
        """
        Initialize dashboard.

        Args:
            config: Dashboard configuration
            port: Server port (if config not provided)
        """
        self.config = config or DashboardConfig(port=port)
        self.api = DashboardAPI()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def connect_hive(self, hive) -> None:
        """Connect to HiveFrame instance."""
        self.api.hive = hive

    def connect_sql(self, sql_context) -> None:
        """Connect to SwarmQL context."""
        self.api.sql_context = sql_context

    def start(self, blocking: bool = True) -> None:
        """
        Start the dashboard server.

        Args:
            blocking: If True, blocks until server stops
        """
        # Set up handler
        DashboardRequestHandler.api = self.api
        DashboardRequestHandler.config = self.config

        self._server = HTTPServer((self.config.host, self.config.port), DashboardRequestHandler)

        self._running = True

        print(f"🐝 Colony Dashboard running at http://{self.config.host}:{self.config.port}")

        if blocking:
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            self._thread = threading.Thread(target=self._server.serve_forever)
            self._thread.daemon = True
            self._thread.start()

    def stop(self) -> None:
        """Stop the dashboard server."""
        self._running = False
        if self._server:
            self._server.shutdown()
            self._server = None
        print("Dashboard stopped")

    @property
    def is_running(self) -> bool:
        """Check if dashboard is running."""
        return self._running


# Command-line interface
def main():
    """Run dashboard from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="HiveFrame Colony Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    config = DashboardConfig(host=args.host, port=args.port, debug=args.debug)

    dashboard = Dashboard(config)
    dashboard.start()


if __name__ == "__main__":
    main()
