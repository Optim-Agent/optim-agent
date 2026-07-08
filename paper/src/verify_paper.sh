#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

score=0
tex=main.tex
pdf=main.pdf
log=main.log

if [[ ! -f "$tex" ]]; then
  echo 0
  exit 0
fi

rm -f main.aux main.bbl main.blg main.log main.out "$pdf"
pdflatex -interaction=nonstopmode -halt-on-error "$tex" >/tmp/optim-agent-paper-pdflatex1.log
bibtex main >/tmp/optim-agent-paper-bibtex.log || true
pdflatex -interaction=nonstopmode -halt-on-error "$tex" >/tmp/optim-agent-paper-pdflatex2.log
pdflatex -interaction=nonstopmode -halt-on-error "$tex" >/tmp/optim-agent-paper-pdflatex3.log

python3 - <<'PY'
from pathlib import Path
import re

tex = Path("main.tex")
log = Path("main.log")
aux = Path("main.aux")
pdf = Path("main.pdf")
text = tex.read_text(errors="ignore")
log_text = log.read_text(errors="ignore") if log.exists() else ""
aux_text = aux.read_text(errors="ignore") if aux.exists() else ""

def has(pattern):
    return re.search(pattern, text, re.I | re.S) is not None

def total_pages():
    match = re.search(r"Output written on .*?\((\d+) pages?", log_text)
    return int(match.group(1)) if match else 0

def references_page():
    match = re.search(
        r"\\newlabel\{sec:references-start\}\{\{.*?\}\{(\d+)\}",
        aux_text,
    )
    return int(match.group(1)) if match else 0

score = 0
pages = total_pages()
ref_page = references_page()
main_pages = ref_page - 1 if ref_page else 0

if pages <= 0 or not pdf.exists():
    print(0)
    raise SystemExit

# AAAI-27 permits 7 pages of main content and 9 total pages. Make page use the
# dominant gate: a short draft cannot score as acceptance-ready.
if pages > 9 or main_pages > 7:
    hard_cap = 20
elif main_pages <= 3:
    hard_cap = 9
elif main_pages == 4:
    hard_cap = 30
elif main_pages == 5:
    hard_cap = 55
elif main_pages == 6:
    hard_cap = 78
else:
    hard_cap = 100

score += 35 if main_pages == 7 and ref_page == 8 else max(0, main_pages) * 4
score += 10 if pages <= 9 else 0
score += 5 if 8 <= ref_page <= 9 else 0

required_sections = [
    r"\\section\{Introduction\}",
    r"\\section\{Method\}",
    r"\\section\{Experiments\}",
    r"\\section\{Discussion\}",
    r"\\section\{Limitations",
    r"\\section\{Conclusion\}",
]
score += 14 if all(has(p) for p in required_sections) else 2 * sum(has(p) for p in required_sections)
score += 5 if "\\begin{abstract}" in text and len(text.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0].split()) >= 120 else 0
score += 5 if "Anonymous Submission" in text and "\\usepackage[submission]{aaai2027}" in text else 0
score += min(8, len(text.split()) // 600)
score += 6 if text.count("\\includegraphics") >= 5 else text.count("\\includegraphics")
score += 6 if text.count("\\begin{table") >= 4 else text.count("\\begin{table")
score += 8 if all(k in text for k in ["Branin", "Ackley", "MNIST", "CIFAR-10", "ablation"]) else sum(k in text for k in ["Branin", "Ackley", "MNIST", "CIFAR-10", "ablation"])
math_signals = [
    r"\\begin\{equation\}",
    r"f_\{\\mathrm\{Branin\}\}",
    r"f_\{\\mathrm\{Ackley\}\}",
    r"\\arg\\min",
    r"\\mathcal\{D\}_t",
    r"\\pi_\\theta",
    r"\\rho_\\theta",
]
score += min(8, sum(1 for p in math_signals if re.search(p, text)))
supplement_signals = ["supplement", "raw JSON logs", "seeds", "trial parameters", "best configuration"]
score += 5 if all(s in text for s in supplement_signals) else sum(s in text for s in supplement_signals)
score += 5 if text.count("\\cite") >= 12 else min(5, text.count("\\cite"))
score += 4 if "Reproducibility" in text and "seeds" in text and "A800" in text and "A100" in text else 0
score += 3 if "Related Work" in text else 0
score += 3 if "Broader Impact" in text or "Ethical Statement" in text else 0

bad = [r"\\usepackage\{hyperref\}", r"\\vspace\{-", r"\\newpage", r"\\clearpage", r"\\resizebox", r"\\input\{"]
score -= 20 if any(re.search(p, text) for p in bad) else 0
score -= 12 if any(s in text for s in ["examples/", "docs/assets", "../", "README", "github", "http://", "https://"]) else 0
score -= 10 if "Undefined control sequence" in log_text or "LaTeX Error" in log_text else 0
score -= 5 if "Overfull \\hbox" in log_text else 0
score += 3 if pdf.stat().st_size > 100_000 else 0

score = max(0, min(100, score, hard_cap))
print(score)
PY
