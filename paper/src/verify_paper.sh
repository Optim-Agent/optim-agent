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
pdf = Path("main.pdf")
text = tex.read_text(errors="ignore")
log_text = log.read_text(errors="ignore") if log.exists() else ""
score = 0

def has(pattern):
    return re.search(pattern, text, re.I | re.S) is not None

required_sections = [
    r"\\section\{Introduction\}",
    r"\\section\{Method\}",
    r"\\section\{Experiments\}",
    r"\\section\{Discussion\}",
    r"\\section\{Limitations",
    r"\\section\{Conclusion\}",
]
score += 20 if all(has(p) for p in required_sections) else 3 * sum(has(p) for p in required_sections)
score += 8 if "\\begin{abstract}" in text and len(text.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0].split()) >= 120 else 0
score += 8 if "Anonymous Submission" in text and "\\usepackage[submission]{aaai2027}" in text else 0
score += min(10, len(text.split()) // 350)
score += 8 if text.count("\\includegraphics") >= 5 else 2 * text.count("\\includegraphics")
score += 8 if text.count("\\begin{table") >= 4 else 2 * text.count("\\begin{table")
score += 10 if all(k in text for k in ["Branin", "Ackley", "MNIST", "CIFAR-10", "ablation"]) else 2 * sum(k in text for k in ["Branin", "Ackley", "MNIST", "CIFAR-10", "ablation"])
score += 8 if all(p in text for p in ["examples/hard_functions.py", "examples/mnist.py", "examples/cifar10.py", "examples/ablations.py"]) else 2 * sum(p in text for p in ["examples/hard_functions.py", "examples/mnist.py", "examples/cifar10.py", "examples/ablations.py"])
score += 8 if text.count("\\cite") >= 12 else min(8, text.count("\\cite"))
score += 5 if "Reproducibility" in text and "seeds" in text and "A800" in text and "A100" in text else 0
score += 5 if "Related Work" in text else 0
score += 5 if "Broader Impact" in text or "Ethical Statement" in text else 0

bad = [r"\\usepackage\{hyperref\}", r"\\vspace\{-", r"\\newpage", r"\\clearpage", r"\\resizebox", r"\\input\{"]
score -= 20 if any(re.search(p, text) for p in bad) else 0
score -= 10 if "Undefined control sequence" in log_text or "LaTeX Error" in log_text else 0
score -= 5 if "Overfull \\hbox" in log_text else 0
score += 5 if pdf.exists() and pdf.stat().st_size > 100_000 else 0

print(max(0, min(100, score)))
PY
