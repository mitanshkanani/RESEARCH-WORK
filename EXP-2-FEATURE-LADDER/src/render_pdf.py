"""Turn JOURNAL.md into EXP-2-JOURNAL.pdf (markdown -> HTML -> headless Chrome or Edge).

SVG figures referenced from the markdown as ![caption](figures/name.svg) are embedded as vectors.
Run:  python src/render_pdf.py
"""
import pathlib
import subprocess
import tempfile

import markdown

from common import ROOT

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: "Segoe UI", system-ui, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #0b0b0b; }
h1 { font-size: 20pt; margin: 0 0 4pt; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; border-bottom: 1px solid #c3c2b7; padding-bottom: 3pt; break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; break-after: avoid; }
table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt; break-inside: avoid; }
th, td { border: 1px solid #e1e0d9; padding: 3pt 7pt; text-align: left; vertical-align: top; }
th { background: #f0efec; }
code { font-family: Consolas, monospace; font-size: 9pt; background: #f0efec; padding: 0 2pt; }
img { width: 100%; margin: 6pt 0 2pt; break-inside: avoid; }
p:has(> img) { break-inside: avoid; }
blockquote { margin: 8pt 0; padding: 6pt 12pt; border-left: 3px solid #2a78d6; background: #f4f8fe; }
"""


def main():
    md = (ROOT / "JOURNAL.md").read_text(encoding="utf-8")
    body = markdown.markdown(md, extensions=["tables", "sane_lists"])
    html = f"<!doctype html><html><head><meta charset='utf-8'><base href='{ROOT.as_uri()}/'>" \
           f"<style>{CSS}</style></head><body>{body}</body></html>"
    src = ROOT / "results" / "journal.html"
    src.write_text(html, encoding="utf-8")
    out = ROOT / "EXP-2-JOURNAL.pdf"
    browsers = ["C:/Program Files/Google/Chrome/Application/chrome.exe",
                "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
                "C:/Program Files/Microsoft/Edge/Application/msedge.exe"]
    exe = next(b for b in browsers if pathlib.Path(b).exists())
    with tempfile.TemporaryDirectory(dir=ROOT / "results") as tmp:
        subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        f"--user-data-dir={tmp}", f"--print-to-pdf={out}", src.as_uri()],
                       check=True, capture_output=True, timeout=120)
    print("wrote", out, f"{out.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
