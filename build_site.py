"""Inline site/data.js into site/src.html -> site/index.html (one self-contained page)."""
from pathlib import Path
root = Path(__file__).parent / "site"
html = (root / "src.html").read_text(encoding="utf-8")
data = (root / "data.js").read_text(encoding="utf-8")
assert "/*__DATA__*/" in html
(root / "index.html").write_text(html.replace("/*__DATA__*/", data), encoding="utf-8")
print("built", (root / "index.html").stat().st_size, "bytes")
