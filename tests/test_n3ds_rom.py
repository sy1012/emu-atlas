"""Azahar's 3DS save unit: the image's program and extdata ids, and the two directories a title owns.

Three layers, each proven on its own: the pure read (:mod:`atlas.n3ds_rom`
over synthetic NCSD and NCCH bytes), the real machine's read over files on
disk, and the save answer over fixture machines. The conformance vectors
(``vectors/machines/azahar-titles.json``) hold the serialized answers.
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
from atlas import n3ds_rom
from atlas.contract import savefile_placement_contract
from atlas.machine import (
    N3DS_ROM_ENCRYPTED,
    N3DS_ROM_FORMAT_NOT_READ,
    N3DS_ROM_MISSING,
    N3DS_ROM_NOT_A_ROM,
    N3DS_ROM_OK,
    FixtureMachine,
    RealMachine,
)
from atlas.placement import SAVE_UNIT_ROOTED_FOLDERS, SaveUnitStatement

PROGRAM = 0x0004000000113200
EXTDATA = 0x1132

# --- Synthetic bytes ---------------------------------------------------------


def ncch(
    program_id: int = PROGRAM,
    *,
    extdata_id: int = EXTDATA,
    accessible: int = 0,
    other_attributes: int = 0,
    no_crypto: bool = True,
    exheader: bool = True,
    content_size: int = 0x10,
) -> bytes:
    """An NCCH header and the extended header up to its storage info, padded to 0x800 bytes."""
    data = bytearray(0x800)
    data[0x100:0x104] = b"NCCH"
    struct.pack_into("<I", data, 0x104, content_size)
    struct.pack_into("<Q", data, 0x118, program_id)
    struct.pack_into("<I", data, 0x180, 0x400 if exheader else 0)
    data[0x18F] = 0x04 if no_crypto else 0x00
    if exheader:
        storage = 0x200 + 0x230
        struct.pack_into("<QQQ", data, storage, extdata_id, 0, accessible)
        data[storage + 0x1F] = other_attributes
    return bytes(data)


def ncsd(program: bytes, *, partition_units: int = 0x4, size_units: int | None = None) -> bytes:
    """A card image whose partition 0 is *program*, at *partition_units* media units."""
    head = bytearray(0x200)
    head[0x100:0x104] = b"NCSD"
    units = size_units if size_units is not None else len(program) // n3ds_rom.MEDIA_UNIT
    struct.pack_into("<II", head, 0x120, partition_units, units)
    gap = partition_units * n3ds_rom.MEDIA_UNIT - len(head)
    return bytes(head) + b"\0" * gap + program


def read(data: bytes) -> n3ds_rom.N3dsRomFacts:
    return n3ds_rom.read_facts(io.BytesIO(data), len(data))


class TestTheImage:
    def test_a_card_image_reads_partition_zeros_program(self):
        facts = read(ncsd(ncch()))
        assert facts == n3ds_rom.N3dsRomFacts("ncsd", PROGRAM, EXTDATA, n3ds_rom.EXTDATA_EXHEADER_ID)

    def test_an_ncch_is_the_program_itself(self):
        facts = read(ncch())
        assert facts.container == "ncch" and facts.program_id == PROGRAM and facts.extdata_id == EXTDATA

    def test_the_save_paths_spell_the_ids(self):
        assert n3ds_rom.title_words(PROGRAM) == ("00040000", "00113200")
        assert n3ds_rom.extdata_words(0xB8B) == ("00000000", "00000B8B")

    def test_extended_access_takes_the_first_non_zero_listed_id(self):
        # id0..id2 ride storage_accessible_unique_ids at bits 40, 20, 0; id3..id5 ext_save_data_id.
        facts = read(ncch(accessible=0xB8B, extdata_id=0x1234 << 40, other_attributes=0x02))
        assert facts.extdata_id == 0xB8B and facts.extdata_basis == n3ds_rom.EXTDATA_EXTENDED_FIRST
        facts = read(ncch(accessible=0, extdata_id=0x1234 << 20, other_attributes=0x02))
        assert facts.extdata_id == 0x1234

    def test_extended_access_listing_nothing_names_no_extdata(self):
        facts = read(ncch(accessible=0, extdata_id=0, other_attributes=0x02))
        assert facts.extdata_id is None and facts.extdata_basis is None

    def test_a_zero_id_names_no_extdata(self):
        assert read(ncch(extdata_id=0)).extdata_id is None

    def test_no_extended_header_names_no_extdata(self):
        assert read(ncch(exheader=False)).extdata_id is None

    def test_a_prototype_reads_its_extended_header_without_a_size(self):
        data = bytearray(ncch())
        struct.pack_into("<I", data, 0x180, 0)
        struct.pack_into("<I", data, 0x104, len(data))
        assert read(bytes(data)).extdata_id == EXTDATA

    def test_an_encrypted_ncch_is_refused(self):
        with pytest.raises(n3ds_rom.N3dsRomEncrypted):
            read(ncsd(ncch(no_crypto=False)))

    @pytest.mark.parametrize(
        "data",
        [b"", b"\0" * 0x104, b"PK\x03\x04" + b"\0" * 0x400, b"\x20\x20\0\0" + b"\0" * 0x2000],
        ids=["empty", "zeros", "zip", "cia-header"],
    )
    def test_bytes_with_no_magic_are_no_format_read(self, data: bytes):
        with pytest.raises(n3ds_rom.N3dsRomFormatNotRead):
            read(data)

    @pytest.mark.parametrize(
        "data",
        [
            ncsd(ncch())[:0x110],
            ncsd(ncch(), partition_units=0),
            ncsd(ncch(), size_units=0),
            ncsd(b"\0" * 0x800),
            ncsd(ncch())[:0x900],
            ncch()[:0x150],
            ncch()[:0x300],
        ],
        ids=[
            "ncsd-short",
            "partition-at-zero",
            "partition-empty",
            "partition-not-ncch",
            "partition-truncated",
            "ncch-header-short",
            "exheader-short",
        ],
    )
    def test_a_structure_running_short_is_unread(self, data: bytes):
        with pytest.raises(n3ds_rom.N3dsRomUnread):
            read(data)

    def test_a_partition_past_the_end_is_unread(self):
        data = ncsd(ncch(), partition_units=0x4)
        with pytest.raises(n3ds_rom.N3dsRomUnread):
            n3ds_rom.read_facts(io.BytesIO(data[:0x200]), 0x200)

    @pytest.mark.parametrize(
        ("container", "program", "extdata", "basis"),
        [
            ("cia", PROGRAM, None, None),
            ("ncsd", -1, None, None),
            ("ncsd", PROGRAM, 0, None),
            ("ncsd", PROGRAM, EXTDATA, None),
            ("ncsd", PROGRAM, None, n3ds_rom.EXTDATA_EXHEADER_ID),
            ("ncsd", PROGRAM, EXTDATA, "guessed"),
        ],
    )
    def test_facts_out_of_shape_are_refused(self, container, program, extdata, basis):
        with pytest.raises(ValueError):
            n3ds_rom.N3dsRomFacts(container, program, extdata, basis)


# --- The real machine ----------------------------------------------------------


class TestTheRealMachine:
    def test_it_reads_an_image_and_its_failures(self, tmp_path: Path):
        good = tmp_path / "Game.3ds"
        good.write_bytes(ncsd(ncch()))
        cia = tmp_path / "Game.cia"
        cia.write_bytes(b"\x20\x20" + b"\0" * 0x3000)
        encrypted = tmp_path / "Enc.3ds"
        encrypted.write_bytes(ncsd(ncch(no_crypto=False)))
        short = tmp_path / "Short.3ds"
        short.write_bytes(ncsd(ncch())[:0x900])
        machine = RealMachine()
        result = machine.read_n3ds_rom(str(good))
        assert result.status == N3DS_ROM_OK and result.facts is not None
        assert result.facts.program_id == PROGRAM
        assert machine.read_n3ds_rom(str(cia)).status == N3DS_ROM_FORMAT_NOT_READ
        assert machine.read_n3ds_rom(str(encrypted)).status == N3DS_ROM_ENCRYPTED
        assert machine.read_n3ds_rom(str(short)).status == N3DS_ROM_NOT_A_ROM
        assert machine.read_n3ds_rom(str(tmp_path / "none.3ds")).status == N3DS_ROM_MISSING
        assert machine.read_n3ds_rom(str(tmp_path)).status == N3DS_ROM_MISSING


# --- The save unit -------------------------------------------------------------


class TestTheStatement:
    def test_a_rooted_unit_states_its_roots(self):
        roots = (("data", "/sd/title/a/b/data"), ("extdata", "/sd/extdata/00000000/00001132"))
        assert SaveUnitStatement(SAVE_UNIT_ROOTED_FOLDERS, None, None, archive_roots=roots).archive_roots == roots

    @pytest.mark.parametrize(
        ("members", "archive_folder", "roots", "message"),
        [
            (None, None, None, "archive_roots are stated exactly"),
            ((), None, (("data", "/d"),), "members are stated exactly"),
            (None, "x", (("data", "/d"),), "top level"),
            (None, None, (), "one or more"),
            (None, None, (("extdata", "/e"), ("data", "/d")), "name order"),
            (None, None, (("data", "/d"), ("data", "/e")), "named once"),
            (None, None, (("..", "/d"),), "no entry name"),
            (None, None, (("data", "relative/d"),), "no absolute path"),
        ],
    )
    def test_a_rooted_unit_is_refused_out_of_shape(self, members, archive_folder, roots, message):
        with pytest.raises(ValueError, match=message):
            SaveUnitStatement(SAVE_UNIT_ROOTED_FOLDERS, members, archive_folder, archive_roots=roots)

    def test_another_kind_carries_no_roots(self):
        with pytest.raises(ValueError, match="archive_roots are stated exactly"):
            SaveUnitStatement(atlas.SAVE_UNIT_FOLDER, None, None, archive_roots=(("data", "/d"),))


_VECTORS = Path(__file__).parent.parent / "vectors" / "machines" / "azahar-titles.json"
_BASE: dict[str, Any] = json.loads(_VECTORS.read_text(encoding="utf-8"))["vectors"][0]["input"]
CCI = "/mnt/sd/retrodeck/roms/n3ds/Game.3ds"
SD = "/mnt/sd/retrodeck/saves/n3ds/azahar/sdmc/Nintendo 3DS/" + "0" * 32 + "/" + "0" * 32


def answer(spec: object, *, content_path: str | None = CCI) -> atlas.SavefilePlacement:
    inp = copy.deepcopy(_BASE)
    machine = FixtureMachine(inp["files"], dirs=inp["dirs"], n3ds_roms={CCI: spec})
    (install,) = atlas.detect(inp["home"], machine)
    entry = install.emulators_for("n3ds", content_path=content_path).entries[0]
    placement = entry.savefile_location(content_path=content_path)
    assert isinstance(placement, atlas.SavefilePlacement)
    return placement


def facts(extdata: str | None = "0000000000001132", basis: str | None = "exheader-id") -> dict[str, Any]:
    return {"container": "ncsd", "program_id": "0004000000113200", "extdata_id": extdata, "extdata_basis": basis}


class TestTheTitlesDirectories:
    def test_a_read_title_states_both_directories_as_one_unit(self):
        placement = answer(facts())
        data, extdata = f"{SD}/title/00040000/00113200/data", f"{SD}/extdata/00000000/00001132"
        assert placement.dir == data and placement.needs == ()
        unit = placement.file_set.groups[0].unit
        assert unit == SaveUnitStatement(
            SAVE_UNIT_ROOTED_FOLDERS, None, None, archive_roots=(("data", data), ("extdata", extdata))
        )
        assert [g.dir for g in placement.file_set.groups] == [data, extdata]
        assert savefile_placement_contract(placement)["file_set"]["groups"][0]["unit"]["archive_roots"] == {
            "data": data,
            "extdata": extdata,
        }

    def test_no_caveat_refuses_what_the_unit_states(self):
        placement = answer(facts())
        dirs = [c.data["dir"] for c in placement.caveats if c.code == atlas.CAVEAT_FILE_NAMES_UNESTABLISHED]
        assert f"{SD}/extdata" not in dirs
        assert all("save_id" not in c.data for c in placement.caveats)

    def test_a_title_granted_no_extdata_states_its_save_alone(self):
        placement = answer(facts(None, None))
        (group,) = placement.file_set.groups
        assert group.unit is not None and [name for name, _ in group.unit.archive_roots or ()] == ["data"]

    def test_extended_access_says_the_id_is_the_first_of_several(self):
        placement = answer(facts("0000000000000b8b", "extended-access-first"))
        assert placement.file_set.groups[1].dir == f"{SD}/extdata/00000000/00000B8B"
        (ext,) = [c for c in placement.caveats if c.data.get("dir") == f"{SD}/extdata/00000000/00000B8B"]
        assert "first of several" in ext.message

    @pytest.mark.parametrize("spec", ["format-not-read", "encrypted", "not-a-rom"])
    def test_an_image_not_read_keeps_the_hole(self, spec: str):
        placement = answer(spec)
        assert placement.needs == (atlas.HOLE_SAVE_ID,)
        assert all(g.unit is None for g in placement.file_set.groups)
        assert any(spec in source for source in placement.sources)

    def test_no_content_path_keeps_the_hole(self):
        placement = answer(facts(), content_path=None)
        assert placement.needs == (atlas.HOLE_SAVE_ID,)
