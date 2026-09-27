"""Read the official ToT Game24 CSV and make a fixed rank-stratified split."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


OFFICIAL_REVISION = "733b009f627f8e5c81c3e5461391d3aa3e0dd18f"
OFFICIAL_SHA256 = "b9f12b3e36d987a3c714c4cef17d89a137d7c59da26532fcfb93b4821d8111b5"
OFFICIAL_URL = (
    "https://raw.githubusercontent.com/princeton-nlp/tree-of-thought-llm/"
    f"{OFFICIAL_REVISION}/src/tot/data/24/24.csv"
)


def dataset_provenance() -> dict:
    return {"url": OFFICIAL_URL, "revision": OFFICIAL_REVISION,
            "sha256": OFFICIAL_SHA256, "verification": "exact original file bytes"}


@dataclass(frozen=True)
class Puzzle:
    index: int  # zero-based row index, matching ToT's task index
    numbers: tuple[int, int, int, int]

    @property
    def key(self) -> tuple[int, int, int, int]:
        return tuple(sorted(self.numbers))


def load_puzzles(path: Path) -> tuple[list[Puzzle], str]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != OFFICIAL_SHA256:
        raise ValueError(
            f"Official Game24 SHA256 mismatch: expected {OFFICIAL_SHA256}, got {digest}. "
            f"Use the unchanged file from {OFFICIAL_URL}"
        )
    decoded = raw.decode("utf-8-sig")
    rows = list(csv.DictReader(decoded.splitlines()))
    puzzles: list[Puzzle] = []
    for index, row in enumerate(rows):
        numbers = tuple(map(int, row["Puzzles"].split()))
        if len(numbers) != 4:
            raise ValueError(f"Invalid puzzle at index {index}")
        puzzles.append(Puzzle(index, numbers))
    if len(puzzles) != 1362:
        raise ValueError(f"Expected the 1362-row official ToT dataset, got {len(puzzles)}")
    if len({puzzle.key for puzzle in puzzles}) != len(puzzles):
        raise ValueError("Duplicate number multisets in official data")
    if puzzles[900].key != (4, 5, 6, 10):
        raise ValueError("Official ToT index 900 does not match the expected puzzle")
    return puzzles, digest


def split_puzzles(puzzles: list[Puzzle], data_sha256: str, *, seed: int) -> dict:
    if len(puzzles) != 1362:
        raise ValueError("Expected 1362 puzzles")
    test = list(range(900, 1000))
    test_set = set(test)
    val: list[int] = []
    for start in range(0, 1362, 100):
        stratum = [index for index in range(start, min(start + 100, 1362))
                   if index not in test_set]
        if not stratum:
            continue
        count = (len(stratum) + 5) // 10  # 10 per full 100, 6 of the last 62
        ranked = sorted(
            stratum,
            key=lambda index: hashlib.sha256(
                f"game24-split-v1:{seed}:{index}:{' '.join(map(str, puzzles[index].key))}".encode()
            ).hexdigest(),
        )
        val.extend(ranked[:count])
    val.sort()
    val_set = set(val)
    train = [index for index in range(1362) if index not in val_set | test_set]
    if (len(train), len(val), len(test)) != (1136, 126, 100):
        raise AssertionError("Unexpected split sizes")
    return {
        "schema_version": 1,
        "source_sha256": data_sha256,
        "split_seed": seed,
        "method": "original-rank blocks of 100; SHA256 ordering within each block",
        "train": train,
        "validation": val,
        "test": test,
    }


def save_split(path: Path, split: dict) -> None:
    path.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
