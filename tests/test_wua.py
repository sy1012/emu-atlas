"""Cemu's Wii U save unit: the title a .wua (or a folder dump) launches, and its save folder whole.

Three layers, each proven on its own: the pure read (:mod:`atlas.wua` over
synthetic ZArchive bytes), the real machine's read over files on disk, and the
save answer over fixture machines. The conformance vectors
(``vectors/machines/cemu-titles.json``) hold the serialized answers.
"""

from __future__ import annotations

import copy
import io
import json
import struct
from pathlib import Path
from typing import Any

import pytest

import atlas
from atlas import wua
from atlas.contract import savefile_placement_contract
from atlas.machine import (
    WUA_TITLES_FORMAT_NOT_READ,
    WUA_TITLES_MISSING,
    WUA_TITLES_NOT_AN_ARCHIVE,
    WUA_TITLES_OK,
    FixtureMachine,
    RealMachine,
)
from atlas.placement import SAVE_UNIT_FOLDER, SaveUnitStatement

BASE = "0005000010101c00"
UPDATE = "0005000e10101c00"
DLC = "0005000c10101c00"

# --- Synthetic bytes ---------------------------------------------------------

Node = dict[str, "Node | None"]  # a name -> a directory's children, or None for a file


def zarchive(tree: Node, *, root_name_offset: int = 0x7FFFFFFF, total_size: int | None = None) -> bytes:
    """An archive with *tree* as its top level, laid out as ZArchiveWriter lays one out.

    No file data is written (every file is empty); the index is the writer's:
    names first, then the file tree breadth-first with each directory's
    children sorted, then the footer.
    """
    names = bytearray()
    offsets: dict[int, int] = {}
    order: list[tuple[str | None, Node | None]] = [(None, tree)]
    entries: list[tuple[int, int, int]] = []
    next_index = 1
    i = 0
    while i < len(order):
        name, node = order[i]
        if name is None:
            name_offset = root_name_offset
        else:
            raw = name.encode("utf-8")
            name_offset = len(names)
            if len(raw) >= 0x80:
                names += bytes([(len(raw) & 0x7F) | 0x80, len(raw) >> 7])
            else:
                names.append(len(raw))
            names += raw
        offsets[i] = name_offset
        if node is None:
            entries.append((0x80000000 | name_offset, 0, 0))
        else:
            children = sorted(node.items(), key=lambda kv: kv[0].lower())
            entries.append((name_offset, next_index, len(children)))
            next_index += len(children)
            order.extend(children)
        i += 1
    records = struct.pack(">Q16H", 0, *([0] * 16))
    tree_bytes = b"".join(struct.pack(">IIII", flag, a, b, 0) for flag, a, b in entries)
    body = bytes(records) + bytes(names) + tree_bytes
    sections = [
        (0, 0),
        (0, len(records)),
        (len(records), len(names)),
        (len(records) + len(names), len(tree_bytes)),
        (len(body), 0),
        (len(body), 0),
    ]
    size = len(body) + wua.FOOTER_SIZE
    footer = b"".join(struct.pack(">QQ", o, s) for o, s in sections) + b"\0" * 32
    footer += struct.pack(">QII", size if total_size is None else total_size, 0x61BF3A01, 0x169F52D6)
    return body + footer


def footer_with(data: bytes, section: int, offset: int, size: int) -> bytes:
    """*data* with section *section* of its footer restated."""
    out = bytearray(data)
    struct.pack_into(">QQ", out, len(data) - wua.FOOTER_SIZE + section * 16, offset, size)
    return bytes(out)


def title(folder: str) -> wua.WuaTitle:
    parsed = wua.parse_title_folder(folder.encode())
    assert parsed is not None
    return parsed


def read(data: bytes) -> wua.WuaFacts:
    return wua.read_facts(io.BytesIO(data), len(data))


GAME: Node = {
    f"{BASE}_v0": {"code": {"app.xml": None}, "meta": {"meta.xml": None}},
    f"{UPDATE}_v208": {"code": {"app.xml": None}},
    f"{DLC}_v16": {"content": {}},
}


class TestTheArchive:
    def test_the_top_level_title_folders_are_read_in_the_archives_order(self):
        facts = read(zarchive(GAME))
        assert [f"{t.title_id:016x}_v{t.version}" for t in facts.titles] == [
            f"{BASE}_v0",
            f"{DLC}_v16",
            f"{UPDATE}_v208",
        ]

    def test_only_the_base_title_is_bootable(self):
        assert [t.bootable for t in read(zarchive(GAME)).titles] == [True, False, False]
        assert title("0005000210101d00_v0").bootable and title("0005001010047000_v0").bootable
        assert title("0005003010047000_v0").bootable

    def test_files_and_other_folders_at_the_top_are_not_titles(self):
        facts = read(zarchive({f"{BASE}_v0.txt": None, "readme": {}, f"{BASE}_v0": {}, f"{BASE}_v1": None}))
        assert facts.titles == (wua.WuaTitle(int(BASE, 16), 0),)

    def test_a_name_with_the_long_header_is_read_and_skipped(self):
        facts = read(zarchive({"x" * 200: {}, f"{BASE}_v0": {}}))
        assert facts.titles == (wua.WuaTitle(int(BASE, 16), 0),)

    def test_an_empty_archive_names_no_title(self):
        assert read(zarchive({})).titles == ()

    def test_the_save_path_spells_the_words_lowercase(self):
        assert wua.title_words(0x0005000010101C00) == ("00050000", "10101c00")

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("0005000010101C00_v32", (0x0005000010101C00, 32)),
            ("0005000010101c00_v0", (0x0005000010101C00, 0)),
            ("0005000010101c00_v65535", (0x0005000010101C00, 65535)),
            ("0005000010101c00_V32", None),
            ("0005000010101c00_v032", None),
            ("0005000010101c00_v65536", None),
            ("0005000010101c00_v3a", None),
            ("0005000010101c00_v", None),
            ("0005000010101c0_v32", None),
            ("0005000010101g00_v32", None),
            ("0005000010101c00-v32", None),
        ],
    )
    def test_a_folder_name_is_read_as_cemu_reads_it(self, name: str, expected):
        parsed = wua.parse_title_folder(name.encode())
        assert (None if parsed is None else (parsed.title_id, parsed.version)) == expected

    def test_a_version_that_wraps_a_u32_is_read_as_the_wrapped_value(self):
        # 4294967296 is 2**32: the u32 accumulator wraps it to 0, which Cemu accepts.
        assert wua.parse_title_folder(b"0005000010101c00_v4294967296") == wua.WuaTitle(0x0005000010101C00, 0)

    @pytest.mark.parametrize(
        "data",
        [
            b"",
            b"\0" * wua.FOOTER_SIZE,
            b"\0" * 0x20000,
            b"WUX0\x99\x10\x12\x10" + b"\0" * 0x20000,
            zarchive(GAME)[:-1],
            zarchive(GAME) + b"\0",
        ],
        ids=["empty", "footer-sized", "zeros", "wux", "truncated", "appended"],
    )
    def test_bytes_with_no_footer_are_no_format_read(self, data: bytes):
        with pytest.raises(wua.WuaFormatNotRead):
            read(data)

    @pytest.mark.parametrize(
        "make",
        [
            lambda: zarchive(GAME, total_size=1),
            lambda: footer_with(zarchive(GAME), 3, 0, 1 << 40),
            lambda: footer_with(zarchive(GAME), 1, 0, 0),
            lambda: footer_with(zarchive(GAME), 1, 0, 39),
            lambda: footer_with(zarchive(GAME), 3, 40, 0),
            lambda: footer_with(zarchive(GAME), 3, 40, 17),
            lambda: zarchive(GAME, root_name_offset=0),
        ],
        ids=[
            "total-size",
            "section-past-the-end",
            "no-offset-records",
            "offset-records-not-whole",
            "empty-tree",
            "tree-not-whole",
            "root-named",
        ],
    )
    def test_a_structure_cemu_refuses_is_unread(self, make):
        with pytest.raises(wua.WuaUnread):
            read(make())

    def test_a_root_that_is_a_file_is_unread(self):
        data = bytearray(zarchive(GAME))
        tree_offset, _ = struct.unpack_from(">QQ", data, len(data) - wua.FOOTER_SIZE + 3 * 16)
        data[tree_offset] |= 0x80
        with pytest.raises(wua.WuaUnread):
            read(bytes(data))

    def test_children_past_the_tree_are_unread(self):
        data = bytearray(zarchive(GAME))
        tree_offset, _ = struct.unpack_from(">QQ", data, len(data) - wua.FOOTER_SIZE + 3 * 16)
        struct.pack_into(">I", data, tree_offset + 8, 1000)
        with pytest.raises(wua.WuaUnread):
            read(bytes(data))

    def test_a_top_level_past_the_bound_is_unread(self):
        data = bytearray(zarchive({}))
        tree_offset, _ = struct.unpack_from(">QQ", data, len(data) - wua.FOOTER_SIZE + 3 * 16)
        struct.pack_into(">I", data, tree_offset + 8, wua.MAX_TOP_LEVEL_ENTRIES + 1)
        with pytest.raises(wua.WuaUnread, match="more than atlas reads"):
            read(bytes(data))

    @pytest.mark.parametrize(("title_id", "version"), [(-1, 0), (1 << 64, 0), (0, -1), (0, 0x10000), (True, 0)])
    def test_a_title_out_of_shape_is_refused(self, title_id, version):
        with pytest.raises(ValueError):
            wua.WuaTitle(title_id, version)


# --- The real machine ----------------------------------------------------------


class TestTheRealMachine:
    def test_it_reads_an_archive_and_its_failures(self, tmp_path: Path):
        good = tmp_path / "Game.wua"
        good.write_bytes(zarchive(GAME))
        cut = tmp_path / "Cut.wua"
        cut.write_bytes(zarchive(GAME)[:-10])
        damaged = tmp_path / "Damaged.wua"
        damaged.write_bytes(zarchive(GAME, total_size=1))
        machine = RealMachine()
        result = machine.read_wua_titles(str(good))
        assert result.status == WUA_TITLES_OK and result.facts is not None
        assert [t.bootable for t in result.facts.titles] == [True, False, False]
        assert machine.read_wua_titles(str(cut)).status == WUA_TITLES_FORMAT_NOT_READ
        assert machine.read_wua_titles(str(damaged)).status == WUA_TITLES_NOT_AN_ARCHIVE
        assert machine.read_wua_titles(str(tmp_path / "none.wua")).status == WUA_TITLES_MISSING
        assert machine.read_wua_titles(str(tmp_path)).status == WUA_TITLES_MISSING


class TestTheFixture:
    def test_a_folder_cemu_does_not_read_as_a_title_is_refused(self):
        with pytest.raises(ValueError, match="no folder name Cemu reads"):
            FixtureMachine({"/r/g.wua": "x"}, wua_archives={"/r/g.wua": {"title_folders": ["readme"]}})

    def test_an_undeclared_archive_is_unreadable(self):
        machine = FixtureMachine({"/r/g.wua": "x"})
        assert machine.read_wua_titles("/r/g.wua").status == "unreadable"


# --- The save answer -------------------------------------------------------------

_VECTORS = Path(__file__).parent.parent / "vectors" / "machines" / "cemu-titles.json"
_BASE: dict[str, Any] = json.loads(_VECTORS.read_text(encoding="utf-8"))["vectors"][0]["input"]
WUA = "/mnt/sd/retrodeck/roms/wiiu/Game.wua"
SAVE = "/mnt/sd/retrodeck/bios/cemu/usr/save"
DUMP = "/mnt/sd/retrodeck/roms/wiiu/Dump"


def answer(
    spec: object = None,
    *,
    content_path: str | None = WUA,
    files: dict[str, str] | None = None,
    dirs: tuple[str, ...] = (),
) -> atlas.SavefilePlacement:
    inp = copy.deepcopy(_BASE)
    inp["files"].update(files or {})
    machine = FixtureMachine(
        inp["files"],
        dirs=[*inp["dirs"], *dirs],
        symlinks=inp.get("symlinks"),
        wua_archives={WUA: spec} if spec is not None else None,
    )
    (install,) = atlas.detect(inp["home"], machine)
    entry = install.emulators_for("wiiu", content_path=content_path).entries[0]
    placement = entry.savefile_location(content_path=content_path)
    assert isinstance(placement, atlas.SavefilePlacement)
    return placement


def folders(*names: str) -> dict[str, list[str]]:
    return {"title_folders": list(names)}


class TestTheTitlesSaveFolder:
    def test_a_read_title_states_its_save_folder_as_its_own_unit(self):
        placement = answer(folders(f"{UPDATE}_v208", f"{BASE}_v0", f"{DLC}_v16"))
        folder = f"{SAVE}/00050000/10101c00"
        assert placement.dir == folder and placement.needs == ()
        assert placement.file_set.groups[0].unit == SaveUnitStatement(SAVE_UNIT_FOLDER, None, "10101c00")
        unit = savefile_placement_contract(placement)["file_set"]["groups"][0]["unit"]
        assert unit["kind"] == "own-folder" and unit["archive_folder"] == "10101c00"
        assert any("10101c00_v0" in source for source in placement.sources)

    def test_no_caveat_asks_for_the_hole(self):
        placement = answer(folders(f"{BASE}_v0"))
        assert all("save_id" not in c.data for c in placement.caveats)

    @pytest.mark.parametrize(
        ("spec", "said"),
        [
            (folders(f"{BASE}_v0", "0005000010102200_v0"), "2 titles"),
            (folders(f"{UPDATE}_v208"), "no title"),
            ("format-not-read", "format-not-read"),
            ("not-an-archive", "not-an-archive"),
        ],
    )
    def test_an_archive_naming_no_one_title_keeps_the_hole(self, spec: object, said: str):
        placement = answer(spec)
        assert placement.needs == (atlas.HOLE_SAVE_ID,)
        assert all(g.unit is None for g in placement.file_set.groups)
        assert any(said in source for source in placement.sources)

    def test_no_content_path_keeps_the_hole(self):
        placement = answer(folders(f"{BASE}_v0"), content_path=None)
        assert placement.needs == (atlas.HOLE_SAVE_ID,)

    @pytest.mark.parametrize("suffix", [".wud", ".WUX", ".iso"])
    def test_a_disc_image_keeps_the_hole(self, suffix: str):
        path = f"/mnt/sd/retrodeck/roms/wiiu/Game{suffix}"
        placement = answer(content_path=path, files={path: "disc"})
        assert placement.needs == (atlas.HOLE_SAVE_ID,)
        assert any("disc image" in source for source in placement.sources)


_APP_XML = '<?xml version="1.0"?>\n<app type="complex"><title_id type="hexBinary" length="8">{}</title_id></app>\n'


class TestAFolderDump:
    def dump(self, app_xml: str | None, *, beside: tuple[str, ...] = ("content", "meta"), code: str = "code"):
        rpx = f"{DUMP}/{code}/Game.rpx"
        files = {rpx: "rpx"}
        if app_xml is not None:
            files[f"{DUMP}/{code}/app.xml"] = app_xml
        return answer(content_path=rpx, files=files, dirs=tuple(f"{DUMP}/{part}" for part in beside))

    def test_app_xml_names_the_title(self):
        placement = self.dump(_APP_XML.format("0005000010101C00"), code="CODE")
        assert placement.dir == f"{SAVE}/00050000/10101c00"
        assert placement.file_set.groups[0].unit == SaveUnitStatement(SAVE_UNIT_FOLDER, None, "10101c00")

    @pytest.mark.parametrize(
        "app_xml",
        [
            None,
            "not xml",
            "<menu><title_id>0005000010101C00</title_id></menu>",
            _APP_XML.format("0005000E10101C00"),
            _APP_XML.format("zz"),
            _APP_XML.format("00050000101010C000"),
        ],
        ids=["missing", "unparseable", "not-app", "update", "not-hex", "too-long"],
    )
    def test_an_app_xml_naming_no_launched_title_keeps_the_hole(self, app_xml: str | None):
        assert self.dump(app_xml).needs == (atlas.HOLE_SAVE_ID,)

    def test_an_rpx_outside_a_dump_runs_under_a_placeholder_and_keeps_the_hole(self):
        placement = self.dump(_APP_XML.format("0005000010101C00"), beside=("content",))
        assert placement.needs == (atlas.HOLE_SAVE_ID,)
        assert any("placeholder" in source for source in placement.sources)
