import webbrowser
import tempfile
from agent import build_agent

mermaid_str = build_agent().get_graph().draw_mermaid()

html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Agent Graph</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
</head>
<body>
  <div class="mermaid">{mermaid_str}</div>
  <script>mermaid.initialize({{startOnLoad: true, theme: 'default'}});</script>
</body>
</html>"""

with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
    f.write(html)
    path = f.name

webbrowser.open(f"file://{path}")
print(f"Graph opened in browser: {path}")
