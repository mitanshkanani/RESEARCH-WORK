"""Render EXP1_JOURNAL.md (or any sibling .md) to PDF via markdown -> styled HTML -> headless Chrome."""
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

MD_EXTENSIONS = ["tables", "fenced_code", "toc", "sane_lists"]

CSS = """
@page { size: A4; margin: 20mm 17mm; }
body { font-family: "Georgia", "Times New Roman", serif; font-size: 10.5pt; line-height: 1.45;
       color: #1a1a1a; max-width: 100%; }
h1 { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 20pt; margin: 0 0 2mm;
     border-bottom: 2.5pt solid #1a1a1a; padding-bottom: 2mm; }
h2 { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 14pt; margin-top: 8mm;
     border-bottom: 0.6pt solid #b3b3b3; padding-bottom: 1mm; }
h3 { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 11.5pt; margin-top: 5mm; color: #333; }
h4 { font-family: "Helvetica Neue", Arial, sans-serif; font-size: 10.5pt; margin-top: 4mm; }
p, li { text-align: justify; }
code { font-family: "Consolas", "Courier New", monospace; font-size: 9pt; background: #f2f2f2;
       padding: 0 1pt; }
pre { background: #f7f7f7; border: 0.5pt solid #d4d4d4; border-left: 2.5pt solid #6a7fb5;
      padding: 3mm; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.5pt; line-height: 1.35; }
table { border-collapse: collapse; width: 100%; margin: 3mm 0; font-size: 9pt;
        font-family: "Helvetica Neue", Arial, sans-serif; page-break-inside: avoid; }
th, td { border: 0.5pt solid #9a9a9a; padding: 1.6mm 2mm; vertical-align: top; text-align: left; }
th { background: #e8eaf0; font-weight: 700; }
tr:nth-child(even) td { background: #fafafa; }
blockquote { border-left: 2.5pt solid #6a7fb5; margin: 3mm 0; padding: 1mm 0 1mm 4mm; color: #444; }
hr { border: none; border-top: 0.5pt solid #cccccc; margin: 6mm 0; }
a { color: #1a3a6b; text-decoration: none; word-break: break-word; }
"""


def build(md_path: Path) -> Path:
    html = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=MD_EXTENSIONS)
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{md_path.stem}</title><style>{CSS}</style></head><body>{html}</body></html>"
    )
    tmp = Path(tempfile.mkdtemp()) / (md_path.stem + ".html")
    tmp.write_text(doc, encoding="utf-8")

    chrome = Path(r"C:/Program Files/Google/Chrome/Application/chrome.exe")
    if not chrome.exists():
        chrome = Path(r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe")
    out = md_path.with_suffix(".pdf")  # Chrome needs an ABSOLUTE path, not a relative one
    profile = tempfile.mkdtemp()
    subprocess.run(
        [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile}",
            f"--print-to-pdf={out}",
            tmp.as_uri(),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if not out.exists():
        sys.exit(f"PDF was not produced. chrome stderr:\n{chrome}")
    return out


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "EXP1_JOURNAL.md"
    print(build(target.resolve()))
