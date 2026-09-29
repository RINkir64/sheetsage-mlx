"""
Interactive HTML sheet music generator using abcjs for SheetSage2 MLX.
"""

from pathlib import Path


def generate_interactive_score_html(abc_text: str, out_path: Path, title: str = "Sheet Music") -> Path:
    """
    Generate an interactive responsive sheet music HTML file using abcjs CDN.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    escaped_abc = abc_text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <script src="https://cdn.jsdelivr.net/npm/abcjs@6.4.4/dist/abcjs-basic-min.js"></script>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 20px;
      background: #fdfdfd;
      color: #333;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      border-bottom: 2px solid #eaeaea;
      padding-bottom: 10px;
    }}
    .btn {{
      background: #2563eb;
      color: white;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      cursor: pointer;
      font-size: 14px;
      transition: background 0.2s;
    }}
    .btn:hover {{ background: #1d4ed8; }}
    #paper {{ background: white; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); border-radius: 8px; }}
    @media print {{
      .header {{ display: none; }}
      body {{ margin: 0; background: white; }}
      #paper {{ box-shadow: none; padding: 0; }}
    }}
  </style>
</head>
<body>
  <div class="header">
    <h2>🎼 {title}</h2>
    <div>
      <button class="btn" onclick="window.print()">🖨️ Print / Save as PDF</button>
    </div>
  </div>
  <div id="paper"></div>
  <script>
    const abcString = `{escaped_abc}`;
    ABCJS.renderAbc("paper", abcString, {{
      responsive: "resize",
      staffwidth: 900,
      scale: 1.1,
      add_classes: true
    }});
  </script>
</body>
</html>
"""
    out_path.write_text(html_content, encoding="utf-8")
    return out_path
