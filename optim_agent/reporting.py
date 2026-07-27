"""Pure-stdlib terminal table rendering for per-trial logging.

`TableReporter` prints one row per finished trial, preceded by a header row that
is reprinted whenever a trial introduces a parameter key not seen before
(define-by-run: columns grow as new params appear). Columns come solely from the
ordered union of the `trial.params` dicts the reporter has already rendered — it
never reads the shared mutable `study.space` — so it is safe to drive under the
optimize log lock. Rows are truncated with an ellipsis to fit the terminal width
(`shutil.get_terminal_size`); there is no wrapping and no in-place redraw, so
each trial appends plain lines.

`format_line` is the classic single-line format extracted from `Study._log`,
used when stdout is not a TTY so piped/CI logs stay greppable.
"""

import shutil

ELLIPSIS = "…"
MISSING = "-"
_SEPARATOR = " | "
_MIN_COL_WIDTH = 4
_FALLBACK_SIZE = (80, 24)


def _truncate(text, width):
    """Fit `text` into `width` chars, cutting with an ellipsis when needed."""
    if len(text) <= width:
        return text
    if width <= len(ELLIPSIS):
        return text[:width]
    return text[: width - len(ELLIPSIS)] + ELLIPSIS


def _fmt_cell(value):
    """Render a cell value; missing/`None` cells (e.g. failed trials) show `-`."""
    if value is None:
        return MISSING
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def format_line(trial, best):
    """Classic one-line per-trial log (extracted from `Study._log`)."""
    if best is not None:
        return (f"[optim-agent] trial {trial.number}: value={trial.value} "
                f"state={trial.state} best={best:.6g}")
    return f"[optim-agent] trial {trial.number}: state={trial.state}"


class TableReporter:
    """Line-separated trial table: trial, value, best, state, one column per param.

    Parameter columns appear in first-seen (declaration) order across the trials
    rendered so far. When a trial carries a new param key, the header row is
    reprinted with the superset of columns before that trial's row.
    """

    FIXED_COLUMNS = ("trial", "value", "best", "state")

    def __init__(self, width=None):
        self._param_keys = []          # ordered union of seen trial.params keys
        self._widths = {}              # column name -> widest content seen so far
        self._header_printed = False
        self._width = width            # pinned width (tests); None → detect per row

    @property
    def columns(self):
        return list(self.FIXED_COLUMNS) + list(self._param_keys)

    def _terminal_width(self):
        if self._width is not None:
            return self._width
        return shutil.get_terminal_size(_FALLBACK_SIZE).columns

    def _cells(self, trial, best):
        cells = {
            "trial": str(trial.number),
            "value": _fmt_cell(trial.value),
            "best": _fmt_cell(best),
            "state": _fmt_cell(trial.state),
        }
        for key in self._param_keys:
            cells[key] = _fmt_cell(trial.params.get(key))
        return cells

    @staticmethod
    def _shrink(widths, budget, floor):
        while sum(widths) > budget:
            widest = max(range(len(widths)), key=lambda i: widths[i])
            if widths[widest] <= floor:
                return  # everything at the floor; second pass may go lower
            widths[widest] -= 1

    def _fit_widths(self, budget):
        """Shrink the recorded column widths to fit `budget` chars of content."""
        widths = [max(self._widths.get(name, len(name)), len(name))
                  for name in self.columns]
        self._shrink(widths, budget, _MIN_COL_WIDTH)
        self._shrink(widths, budget, 1)
        return widths

    def _render(self, values, width):
        widths = self._fit_widths(width - len(_SEPARATOR) * (len(self.columns) - 1))
        cells = [_truncate(str(v), w).ljust(w) for v, w in zip(values, widths)]
        return _SEPARATOR.join(cells).rstrip()

    def header_line(self):
        return self._render(self.columns, self._terminal_width())

    def row_line(self, trial, best):
        cells = self._cells(trial, best)
        return self._render([cells[name] for name in self.columns],
                            self._terminal_width())

    def report(self, trial, best):
        """Print the row for a finished trial (plus a header when columns grow)."""
        new_keys = [k for k in trial.params if k not in self._param_keys]
        for key in new_keys:
            self._param_keys.append(key)
            self._widths.setdefault(key, len(str(key)))
        cells = self._cells(trial, best)
        for name, cell in cells.items():
            self._widths[name] = max(self._widths.get(name, len(name)), len(cell))
        if new_keys or not self._header_printed:
            print(self.header_line())
            self._header_printed = True
        print(self.row_line(trial, best))
