"""Per-trial console rendering: classic line format and stdlib table format.

``format_line`` is the original one-line-per-trial output, extracted from
``Study._log`` so the line and table verbose modes share one implementation.

``TableReporter`` appends one row per finished trial: trial #, value,
best-so-far, state, plus one column per search-space parameter. Columns are
derived from the ordered union of ``trial.params`` keys across the rows the
reporter has already rendered (first-seen order); it never reads
``study.space``, which define-by-run mutates outside the log lock. When a
trial introduces a new parameter key, the header row reprints with the
superset of columns. Rendering is pure stdlib: width comes from
``shutil.get_terminal_size``, long cells are truncated with an ellipsis to
fit, there is no wrapping and no in-place redraw, so piped logs and CI
output stay intact.
"""

import shutil

ELLIPSIS = "…"
MIN_COL_WIDTH = 3  # floor per column: enough for two chars + ellipsis
_SEPARATOR = "  "
FIXED_HEADERS = ("#", "value", "best", "state")


def _format_number(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_cell(value):
    """Missing values (failed trials, absent params) render as ``-``."""
    if value is None:
        return "-"
    return _format_number(value)


def _truncate(text, width):
    if len(text) <= width:
        return text
    if width <= 0:
        return ""
    if width <= len(ELLIPSIS):
        return ELLIPSIS[:width]
    return text[: width - len(ELLIPSIS)] + ELLIPSIS


def _fit_widths(naturals, total):
    """Shrink the widest column one char at a time until the row fits."""
    budget = total - len(_SEPARATOR) * (len(naturals) - 1)
    widths = list(naturals)
    while sum(widths) > budget:
        widest = max(range(len(widths)), key=lambda i: widths[i])
        if widths[widest] <= MIN_COL_WIDTH:
            break
        widths[widest] -= 1
    return widths


def format_line(trial, best):
    """The classic ``[optim-agent] trial N: ...`` line (one per trial)."""
    if best is None:
        return f"[optim-agent] trial {trial.number}: state={trial.state}"
    return (f"[optim-agent] trial {trial.number}: value={trial.value} "
            f"state={trial.state} best={best:.6g}")


class TableReporter:
    """Append-only trial table: header plus one row per trial, no redraw.

    Pass ``width`` to override the terminal width (mainly for tests); by
    default ``shutil.get_terminal_size().columns`` is consulted on every
    rendered line. Call :meth:`log` under the same lock that serializes
    trial bookkeeping so rows from concurrent workers never interleave.
    """

    def __init__(self, width=None):
        self._width = width
        self._param_keys = []  # ordered union of seen trial.params keys
        self._col_widths = [len(h) for h in FIXED_HEADERS]
        self._header_pending = True

    @property
    def param_keys(self):
        """Parameter columns known so far, in first-seen order."""
        return tuple(self._param_keys)

    def _term_width(self):
        if self._width is not None:
            return self._width
        return shutil.get_terminal_size().columns

    def _render(self, cells, fitted):
        parts = [_truncate(str(c), w).ljust(w) for c, w in zip(cells, fitted)]
        return _SEPARATOR.join(parts).rstrip()

    def log(self, trial, best):
        """Print the header if needed, then one row for ``trial``."""
        new_keys = [k for k in trial.params if k not in self._param_keys]
        if new_keys:
            self._param_keys.extend(new_keys)
            self._col_widths.extend(len(k) for k in new_keys)
            self._header_pending = True
        cells = [str(trial.number), _format_cell(trial.value), _format_cell(best),
                 trial.state]
        cells.extend(_format_cell(trial.params.get(k)) for k in self._param_keys)
        self._col_widths = [max(w, len(c)) for w, c in zip(self._col_widths, cells)]
        fitted = _fit_widths(self._col_widths, self._term_width())
        if self._header_pending:
            headers = list(FIXED_HEADERS) + list(self._param_keys)
            print(self._render(headers, fitted))
            self._header_pending = False
        print(self._render(cells, fitted))
