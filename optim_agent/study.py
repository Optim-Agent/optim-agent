"""Study and Trial: define-by-run optimization loop with JSON or SQLite persistence."""

import json
import os
import random
import sqlite3
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import agent as _agent
from . import space as _space
from . import summarizer as _summarizer
from .reporting import TableReporter, format_line
from .samplers import AgentSampler, RandomSampler


class TrialPruned(Exception):
    """Raise inside an objective (or via trial.should_prune()) to stop the trial early."""


class _SQLiteStore:
    """Concurrency-safe trial history in SQLite (stdlib). Point several processes
    at one file to distribute a study: WAL mode plus a busy timeout let them read
    history and append results without clobbering each other — the DB *is* the
    inter-process communication channel.

    ponytail: one connection per call — fine when trials cost seconds+; pool the
    connection if you ever run thousands of sub-second trials.
    """

    SUFFIXES = (".db", ".sqlite", ".sqlite3")

    def __init__(self, path, direction):
        self.path = str(path)
        c = self._conn()
        try:
            c.execute("CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS space(name TEXT PRIMARY KEY, dist TEXT)")
            c.execute("CREATE TABLE IF NOT EXISTS trials("
                      "number INTEGER PRIMARY KEY AUTOINCREMENT, params TEXT, "
                      "value REAL, state TEXT, intermediate TEXT)")
            got = c.execute("SELECT v FROM meta WHERE k='direction'").fetchone()
            if got is None:
                c.execute("INSERT INTO meta VALUES('direction', ?)", (direction,))
            elif got[0] != direction:
                raise ValueError(f"study at {self.path} was created with "
                                 f"direction={got[0]!r}, got {direction!r}")
            c.commit()
        finally:
            c.close()

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=30)
        c.execute("PRAGMA journal_mode=WAL")    # readers don't block the one writer
        c.execute("PRAGMA busy_timeout=30000")  # wait out a peer's write, don't error
        return c

    def set_meta(self, key, value):
        c = self._conn()
        try:
            c.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, value))
            c.commit()
        finally:
            c.close()

    def get_meta(self, key):
        c = self._conn()
        try:
            row = c.execute("SELECT v FROM meta WHERE k=?", (key,)).fetchone()
            return row[0] if row else None
        finally:
            c.close()

    def add_trial(self, params, value, state, intermediate) -> int:
        c = self._conn()
        try:  # AUTOINCREMENT hands out a unique number even across processes
            cur = c.execute("INSERT INTO trials(params, value, state, intermediate) "
                            "VALUES(?,?,?,?)",
                            (json.dumps(params), value, state, json.dumps(intermediate)))
            c.commit()
            return cur.lastrowid
        finally:
            c.close()

    def save_space(self, space):
        c = self._conn()
        try:
            c.executemany("INSERT OR REPLACE INTO space VALUES(?,?)",
                          [(n, json.dumps(_space.to_dict(d))) for n, d in space.items()])
            c.commit()
        finally:
            c.close()

    def load_space(self):
        c = self._conn()
        try:
            return {n: _space.from_dict(json.loads(d))
                    for n, d in c.execute("SELECT name, dist FROM space").fetchall()}
        finally:
            c.close()

    def all_trials(self):
        c = self._conn()
        try:
            return c.execute("SELECT number, params, value, state, intermediate "
                             "FROM trials ORDER BY number").fetchall()
        finally:
            c.close()


class Trial:
    def __init__(self, study, number, proposal=None, params=None):
        self._study = study
        self.number = number
        self._proposal = proposal or {}
        self.params = dict(params) if params else {}
        self.value = None
        self.state = "running"
        self.intermediate = []  # [(step, value), ...]

    def _suggest(self, name, dist):
        known = self._study.space.get(name)
        if known is None:
            known = self._study.space[name] = dist
        elif known != dist:
            raise ValueError(f"{name!r} was already registered as {known}, got {dist}; "
                             "changing a distribution mid-study invalidates the history")
        if name in self.params:  # explicit ask(params=...) — validate, don't trust
            self.params[name] = known.validate(self.params[name])
            return self.params[name]
        if name in self._proposal:
            try:
                value = known.validate(self._proposal[name])
            except (ValueError, TypeError):
                value = known.sample(self._study._rng)
        else:
            value = known.sample(self._study._rng)
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, *, context=None, log=False):
        return self._suggest(name, _space.Float(low, high, log, context))

    def suggest_int(self, name, low, high, *, context=None, log=False):
        return self._suggest(name, _space.Int(low, high, log, context))

    def suggest_categorical(self, name, choices, *, context=None):
        return self._suggest(name, _space.Categorical(tuple(choices), context))

    def report(self, value, step):
        self.intermediate.append((step, float(value)))

    def should_prune(self) -> bool:
        pruner = self._study.pruner
        return bool(pruner and pruner.should_prune(self._study, self))


class Study:
    def __init__(self, direction="minimize", sampler=None, pruner=None,
                 storage=None, seed=None, max_concurrency=1, summarize=False,
                 summary_backend=None, summary_model=None, summary_effort=None):
        if direction not in ("minimize", "maximize"):
            raise ValueError("direction must be 'minimize' or 'maximize'")
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        self.direction = direction
        self.sampler = sampler or RandomSampler()
        self.pruner = pruner
        self.max_concurrency = max_concurrency
        # fail fast: an unresolvable summary config raises here, not post-optimize
        self._resolve_summary_agent(summarize, summary_backend,
                                    summary_model, summary_effort)
        self.storage = Path(storage) if storage else None
        self._store = None
        self._rng = random.Random(seed)
        self.trials = []
        self.space = {}
        self.summary = None  # LLM-authored narration of the finished run (opt-in)
        if self.storage and str(self.storage).endswith(_SQLiteStore.SUFFIXES):
            self._store = _SQLiteStore(self.storage, direction)
            self.space = self._store.load_space()
            self._refresh_from_store()
            self.summary = self._store.get_meta("summary")  # additive; None when absent
        elif self.storage and self.storage.exists():
            self._load(direction)
        if seed is not None and self.trials:  # don't replay the pre-resume random sequence
            self._rng = random.Random(f"{seed}-{len(self.trials)}")  # str seed: valid on 3.11+, tuple isn't

    def _resolve_summary_agent(self, summarize, backend, model, effort):
        """Resolve the summary-agent config once, at construction (R2).

        Backend: explicit `summary_backend` → the sampler's backend when it is an
        `AgentSampler` → clear ValueError. `summary_model`/`summary_effort`
        override the sampler's model/effort; without an AgentSampler they default
        to None (backend CLI default) and "high" (AgentSampler's default).
        """
        self.summarize = bool(summarize)
        self._summary_backend = self._summary_model = self._summary_effort = None
        if not self.summarize:
            return
        if backend is None:
            if isinstance(self.sampler, AgentSampler):
                backend = self.sampler.backend
            else:
                raise ValueError(
                    "summarize=True needs an agent backend but the sampler is not an "
                    "AgentSampler; pass summary_backend=... "
                    f"(one of {_agent.BACKENDS + ('mock',)})")
        elif backend != "mock" and backend not in _agent.BACKENDS:
            raise ValueError(f"summary_backend must be one of {_agent.BACKENDS + ('mock',)}")
        if model is None and isinstance(self.sampler, AgentSampler):
            model = self.sampler.model
        if effort is None:
            effort = (self.sampler.effort if isinstance(self.sampler, AgentSampler)
                      else "high")
        self._summary_backend = backend
        self._summary_model = model
        self._summary_effort = effort

    # -- ask / tell ----------------------------------------------------------

    def ask(self, params=None) -> Trial:
        """Start a trial. Pass explicit `params` to choose the point yourself
        (skill mode: the agent driving this session is the sampler)."""
        if params:  # unwrap numpy scalars etc. so storage stays a faithful round-trip
            params = {k: v.item() if hasattr(v, "item") else v for k, v in params.items()}
        if self._store is not None and not params:
            self._refresh_from_store()  # see peers' completed trials before proposing
        proposal = dict(params) if params else self.sampler.propose(self)
        number = self.trials[-1].number + 1 if self.trials else 0
        trial = Trial(self, number, proposal, params=params)
        if self._store is None:  # SQLite assigns the real number at tell() time
            self.trials.append(trial)
        return trial

    def tell(self, trial, value=None, state="complete"):
        if state not in ("complete", "pruned", "failed"):
            raise ValueError("state must be 'complete', 'pruned' or 'failed'")
        if value is not None:
            trial.value = float(value)
        elif trial.intermediate:
            trial.value = trial.intermediate[-1][1]
        elif state == "complete":
            raise ValueError("a complete trial needs a value (or reported intermediates)")
        trial.state = state
        if self._store is not None:
            trial.number = self._store.add_trial(trial.params, trial.value, state,
                                                 trial.intermediate)
            self._store.save_space(self.space)
            self.trials.append(trial)  # keep this process's view current between refreshes
        elif self.storage:
            self._save()

    def _refresh_from_store(self):
        self.trials = []
        for number, params, value, state, intermediate in self._store.all_trials():
            t = Trial(self, number, params=json.loads(params))
            t.value, t.state = value, state
            t.intermediate = [tuple(p) for p in json.loads(intermediate)]
            self.trials.append(t)

    def optimize(self, objective, n_trials, catch=(), verbose=True):
        # ponytail: threads, not processes — the objective is an arbitrary (unpicklable)
        # closure whose heavy work (agent subprocess, model training) releases the GIL.
        # For true multi-process, run several processes against one SQLite `storage`.
        mode = self._resolve_verbose(verbose)
        reporter = TableReporter() if mode == "table" else None
        lock = threading.Lock()  # serializes ask/tell; agent queries queue on this lock

        def run_one(_):
            with lock:
                trial = self.ask()
            try:
                value = objective(trial)
            except TrialPruned:
                with lock:
                    self.tell(trial, state="pruned")
            except catch:
                with lock:
                    self.tell(trial, state="failed")
            else:
                with lock:
                    self.tell(trial, value)
            if mode is not None:
                with lock:  # row printing + reporter bookkeeping share this lock (C3)
                    if reporter is not None:
                        reporter.report(trial, self.best_value)
                    else:
                        self._log(trial)
            return trial

        if self.max_concurrency == 1:
            for i in range(n_trials):
                run_one(i)
        else:
            with ThreadPoolExecutor(max_workers=self.max_concurrency) as ex:
                list(ex.map(run_one, range(n_trials)))  # list() re-raises worker errors
        if self._store is not None:
            self._refresh_from_store()
        if self.summarize:
            self._run_summary()
        return self

    @staticmethod
    def _resolve_verbose(verbose):
        """Map `verbose` to "table" | "line" | None (D7): True → "table"; a non-TTY
        stdout (piped/CI) falls back to the greppable line format (R8)."""
        if verbose is True:
            verbose = "table"
        if verbose in (None, False):
            return None
        if verbose == "table":
            return "table" if sys.stdout.isatty() else "line"
        if verbose == "line":
            return "line"
        raise ValueError("verbose must be True, False, 'table' or 'line'")

    def _run_summary(self):
        text = _summarizer.run_summary(self, self._summary_backend,
                                       self._summary_model, self._summary_effort)
        if text is None:  # no complete trials or the agent call failed (R14)
            return
        self.summary = text
        if self._store is not None:
            self._store.set_meta("summary", text)
        elif self.storage:
            self._save()

    def _log(self, trial):
        print(format_line(trial, self.best_value))

    # -- results ---------------------------------------------------------

    @property
    def best_trial(self):
        done = [t for t in self.trials if t.state == "complete" and t.value is not None]
        if not done:
            return None
        pick = min if self.direction == "minimize" else max
        return pick(done, key=lambda t: t.value)

    @property
    def best_value(self):
        t = self.best_trial
        return t.value if t else None

    @property
    def best_params(self):
        t = self.best_trial
        return t.params if t else None

    # -- persistence -------------------------------------------------------

    def _save(self):
        data = {
            "direction": self.direction,
            "summary": self.summary,  # additive key; old readers ignore it
            "space": {n: _space.to_dict(d) for n, d in self.space.items()},
            "trials": [{"number": t.number, "params": t.params, "value": t.value,
                        "state": t.state, "intermediate": t.intermediate}
                       for t in self.trials if t.state != "running"],
        }
        tmp = self.storage.with_suffix(".tmp")  # atomic: a crash mid-write can't eat the study
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, self.storage)

    def _load(self, direction):
        data = json.loads(self.storage.read_text())
        if data["direction"] != direction:
            raise ValueError(f"study at {self.storage} was created with "
                             f"direction={data['direction']!r}, got {direction!r}")
        self.summary = data.get("summary")  # additive key: old studies load as None
        self.space = {n: _space.from_dict(d) for n, d in data["space"].items()}
        for rec in data["trials"]:
            t = Trial(self, rec["number"], params=rec["params"])
            t.value, t.state = rec["value"], rec["state"]
            t.intermediate = [tuple(p) for p in rec["intermediate"]]
            self.trials.append(t)


def create_study(direction="minimize", sampler=None, pruner=None,
                 storage=None, seed=None, max_concurrency=1, summarize=False,
                 summary_backend=None, summary_model=None, summary_effort=None) -> Study:
    """Create a Study.

    summarize: opt-in; when truthy, optimize() runs the summary agent once after
               the last trial, prints the summary and persists it on the study
    summary_backend: agent backend for the summary; defaults to the sampler's
               backend when the sampler is an AgentSampler, else a ValueError
               names this argument as the fix (resolution happens here, R2)
    summary_model / summary_effort: override the sampler's model/effort for the
               summary agent when given
    """
    return Study(direction, sampler, pruner, storage, seed, max_concurrency,
                 summarize=summarize, summary_backend=summary_backend,
                 summary_model=summary_model, summary_effort=summary_effort)
