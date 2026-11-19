#!/usr/bin/env python3
"""Generate atlas/data/pcsx2_memcard_filters.json from PCSX2's GameIndex.yaml.

PCSX2 shows a running game only the folders of a folder memory card whose names
contain its filter: the disc's serial, or — where the game's GameDB entry lists
``memcardFilters`` — those strings instead (``VMManager::UpdateDiscDetails``,
pcsx2/VMManager.cpp:1113 and :1161; ``FilterMatches``,
pcsx2/SIO/Memcard/MemoryCardFolder.cpp:424-446). Which serials carry such a
list is world knowledge the machine does not state in a form atlas reads, so it
is packaged: every GameDB entry with ``memcardFilters``, keyed by its serial as
written, its filters in order.

Nothing here touches the network. Fetch the file at the revision the save card
pins first::

    curl -fLo GameIndex.yaml \\
      https://raw.githubusercontent.com/PCSX2/pcsx2/<revision>/bin/resources/GameIndex.yaml
    python scripts/generate_pcsx2_memcard_filters.py GameIndex.yaml --revision <revision>

The YAML is read by line rather than parsed, because the stdlib has no YAML
reader: a top-level key opens an entry, a ``memcardFilters:`` key two spaces in
opens its list, and each ``- item`` four spaces in is one filter, quoted or
not, a trailing comment cut. The run refuses any ``memcardFilters`` line it
cannot read that way, so a format change upstream stops the run instead of
dropping an entry.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "atlas" / "data" / "pcsx2_memcard_filters.json"

_TOP_KEY = re.compile(r'^([^\s#"][^:]*):\s*(#.*)?$')
_FILTERS_KEY = re.compile(r"^  memcardFilters:\s*(#.*)?$")
_ITEM = re.compile(r'^    - (?:"([^"]*)"|([^\s#"][^#]*?))\s*(#.*)?$')


def _continues(line: str, current: list[str]) -> bool:
    """Whether *line* belongs to the open list: an item, appended, or a blank or comment line passed over."""
    match = _ITEM.match(line)
    if match is not None:
        current.append(match.group(1) if match.group(1) is not None else match.group(2))
        return True
    return not line.strip() or line.lstrip().startswith("#")


def _opens_list(line: str, number: int, serial: str | None) -> str | None:
    """The serial whose ``memcardFilters`` list *line* opens — ``None`` where it opens none; a line naming the key
    any other way stops the run."""
    if "memcardFilters" not in line:
        return None
    if not _FILTERS_KEY.match(line) or serial is None:
        raise SystemExit(f"line {number}: a memcardFilters line this reader does not take: {line!r}")
    return serial


def parse(text: str) -> dict[str, list[str]]:
    """Every entry's ``memcardFilters``, keyed by the serial as written — the first entry where one repeats.

    ``GameDatabase::initDatabase`` skips a serial already loaded, case aside, so
    a repeat's list is read and dropped.
    """
    filters: dict[str, list[str]] = {}
    serial: str | None = None
    current: list[str] | None = None
    for number, line in enumerate(text.splitlines(), 1):
        if current is not None and _continues(line, current):
            continue
        current = None
        top = _TOP_KEY.match(line)
        if top:
            serial = top.group(1)
            continue
        opened = _opens_list(line, number, serial)
        if opened is not None:
            current = []
            if opened.lower() not in {key.lower() for key in filters}:
                filters[opened] = current
    empty = [key for key, value in filters.items() if not value]
    if empty:
        raise SystemExit(f"memcardFilters with no item read: {empty}")
    return filters


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate atlas/data/pcsx2_memcard_filters.json from GameIndex.yaml.")
    parser.add_argument("game_index", type=Path)
    parser.add_argument("--revision", required=True, help="the PCSX2 commit the file was fetched at")
    parser.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    filters = parse(args.game_index.read_text(encoding="utf-8"))
    table = {
        "_meta": {
            "generated_from": "PCSX2 bin/resources/GameIndex.yaml, each entry's memcardFilters",
            "revision": args.revision,
            "entries": len(filters),
        },
        "filters": filters,
    }
    args.output.write_text(json.dumps(table, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
