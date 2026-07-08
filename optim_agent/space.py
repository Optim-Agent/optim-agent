"""Search-space distributions. A study's space is a dict of name -> distribution."""

import math
import random
from dataclasses import dataclass


def _with_context(base: str, context) -> str:
    return base + (f" — {context}" if context else "")


@dataclass
class Float:
    low: float
    high: float
    log: bool = False
    context: str = None  # free-text meaning of this param, surfaced to the agent

    def __post_init__(self):
        if self.log and self.low <= 0:
            raise ValueError(f"log scale requires low > 0, got low={self.low}")
        if self.low > self.high:
            raise ValueError(f"low={self.low} > high={self.high}")

    def describe(self) -> str:
        base = f"float in [{self.low:g}, {self.high:g}]" + (", log scale" if self.log else "")
        return _with_context(base, self.context)

    def sample(self, rng: random.Random) -> float:
        if self.log:
            return math.exp(rng.uniform(math.log(self.low), math.log(self.high)))
        return rng.uniform(self.low, self.high)

    def validate(self, value) -> float:
        v = float(value)  # raises on garbage — that's the point
        if not math.isfinite(v):
            raise ValueError(f"non-finite value {v!r}")
        return min(max(v, self.low), self.high)


@dataclass
class Int:
    low: int
    high: int
    log: bool = False
    context: str = None

    __post_init__ = Float.__post_init__

    def describe(self) -> str:
        base = f"int in [{self.low}, {self.high}]" + (", log scale" if self.log else "")
        return _with_context(base, self.context)

    def sample(self, rng: random.Random) -> int:
        if self.log:
            return round(math.exp(rng.uniform(math.log(self.low), math.log(self.high))))
        return rng.randint(self.low, self.high)

    def validate(self, value) -> int:
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"non-finite value {v!r}")
        return min(max(round(v), self.low), self.high)


@dataclass
class Categorical:
    choices: tuple
    context: str = None

    def __post_init__(self):
        self.choices = tuple(self.choices)

    def describe(self) -> str:
        return _with_context("one of " + repr(list(self.choices)), self.context)

    def sample(self, rng: random.Random):
        return rng.choice(self.choices)

    def validate(self, value):
        for choice in self.choices:  # return the canonical choice, not the agent's spelling of it
            if value == choice:
                return choice
        raise ValueError(f"{value!r} not in {self.choices}")


def to_dict(dist) -> dict:
    d = {"type": type(dist).__name__.lower()}
    d.update(vars(dist))
    if "choices" in d:
        d["choices"] = list(d["choices"])
    if d.get("context") is None:
        d.pop("context", None)  # keep JSON tidy for the common no-context case
    return d


def from_dict(d: dict):
    d = dict(d)
    kind = d.pop("type")
    return {"float": Float, "int": Int, "categorical": Categorical}[kind](**d)
