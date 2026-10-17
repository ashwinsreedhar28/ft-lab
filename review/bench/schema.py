"""Row and decision schemas for review-bench. Pure data; no I/O."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

Kind = Literal["bug", "clean"]
Decision = Literal["approve", "block", "escalate"]


@dataclass
class Hunk:
    file: str
    line_start: int
    line_end: int


@dataclass
class Task:
    id: str
    repo: str
    kind: Kind
    base_sha: str            # parent of the PR
    head_sha: str            # the PR's commit
    diff: str
    files: list[str]
    split: str               # fresh | own-older
    oracle: list[str] = field(default_factory=list)       # FAIL_TO_PASS test ids (bug rows)
    gold_hunks: list[Hunk] = field(default_factory=list)  # where the defect lives (bug rows)
    neighborhood: str = ""   # two-hop context graph serialization
    image: str = ""          # docker image tag for diff+run
    test_cmd: str = ""
    subdir: str = "."        # where test_cmd runs inside the checkout
    fix_sha: str = ""        # bug rows: the commit that fixed it (scoring-only, never shown)

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class Comment:
    file: str
    line_start: int
    line_end: int
    body: str
    proposed_test: str | None = None   # pytest source; graded by running it


@dataclass
class Review:
    decision: Decision
    comments: list[Comment] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    usd: float = 0.0


def overlaps(c: Comment, h: Hunk, slack: int = 3) -> bool:
    return c.file == h.file and c.line_start <= h.line_end + slack and c.line_end >= h.line_start - slack
