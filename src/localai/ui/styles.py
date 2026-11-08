"""Theme and custom styling for LocalAI workbench UI."""

THEME_CSS = """
<style>
/* Modern, dark workbench aesthetics */
.stApp {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #111418;
    border-right: 1px solid #23272e;
}

/* Card container */
.workbench-card {
    background-color: #1a1e24;
    border: 1px solid #2b313a;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 16px;
}

.workbench-card h3 {
    margin-top: 0;
    margin-bottom: 12px;
    font-size: 1.1rem;
    color: #e6edf3;
    font-weight: 600;
}

/* Metric pills */
.metric-pill {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 4px;
}

.pill-success {
    background-color: rgba(46, 160, 67, 0.15);
    color: #3fb950;
    border: 1px solid rgba(46, 160, 67, 0.3);
}

.pill-info {
    background-color: rgba(56, 139, 253, 0.15);
    color: #58a6ff;
    border: 1px solid rgba(56, 139, 253, 0.3);
}

.pill-warning {
    background-color: rgba(210, 153, 34, 0.15);
    color: #d29922;
    border: 1px solid rgba(210, 153, 34, 0.3);
}

/* Citation box */
.citation-box {
    background-color: #161b22;
    border-left: 3px solid #58a6ff;
    padding: 10px 14px;
    margin: 8px 0;
    border-radius: 0 6px 6px 0;
    font-size: 0.88rem;
}

/* Code and JSON output blocks */
pre, code {
    font-family: "Fira Code", "JetBrains Mono", Consolas, monospace !important;
}
</style>
"""
