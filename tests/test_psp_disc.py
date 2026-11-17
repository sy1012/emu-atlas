"""PPSSPP's PSP save unit: the disc's id, a savedata folder's PARAM.SFO, and the folders a game owns.

Three layers, each proven on its own: the pure reads (:mod:`atlas.psp_disc`
over synthetic ISO and PARAM.SFO bytes), the real machine's two reads over
files on disk, and the save answer over fixture machines. The conformance
vectors (``vectors/machines/ppsspp-savedata.json``) hold the serialized
answers.
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
from atlas import psp_disc
from atlas.machine import (
    PSP_DISC_FORMAT_NOT_READ,
    PSP_DISC_MISSING,
    PSP_DISC_NOT_A_DISC,
    PSP_DISC_OK,
    PSP_PARAM_SFO_MISSING,
    PSP_PARAM_SFO_NOT_AN_SFO,
    PSP_PARAM_SFO_OK,
    FixtureMachine,
    RealMachine,
)
from atlas.placement import SAVE_UNIT_PREFIXED_FOLDERS, SaveUnitStatement

BLOCK = psp_disc.BLOCK_SIZE


# --- Synthetic bytes ---------------------------------------------------------


def sfo(entries: dict[str, str | int | bytes]) -> bytes:
    """A PARAM.SFO holding *entries*: strings as 0x0204, ints as 0x0404, bytes as 0x0004."""
    keys = b""
    data = b""
    index = b""
    for key, value in entries.items():
        if isinstance(value, str):
            raw = value.encode() + b"\0"
            fmt, length, max_length = 0x0204, len(raw), len(raw) + 3 & ~3
            raw = raw.ljust(max_length, b"\0")
        elif isinstance(value, int):
            raw, fmt, length, max_length = struct.pack("<I", value), 0x0404, 4, 4
        else:
            raw, fmt, length, max_length = value, 0x0004, len(value), len(value)
        index += struct.pack("<HHIII", len(keys), fmt, length, max_length, len(data))
        keys += key.encode() + b"\0"
        data += raw
    key_start = 20 + len(index)
    keys = keys.ljust(len(keys) + 3 & ~3, b"\0")
    data_start = key_start + len(keys)
    header = struct.pack("<IIIII", 0x46535000, 0x101, key_start, data_start, len(entries))
    return header + index + keys + data


def _record(name: bytes, lba: int, size: int, *, directory: bool) -> bytes:
    body = struct.pack("<BBI4sI4s7sBBBHH", 0, 0, lba, b"\0" * 4, size, b"\0" * 4, b"\0" * 7,
                       0x02 if directory else 0, 0, 0, 1, 0)
    record = body[:32] + bytes([len(name)]) + name
    if len(record) % 2:
        record += b"\0"
    return bytes([len(record)]) + record[1:]


def iso(param_sfo: bytes | None, *, game_dir: bytes = b"PSP_GAME", sfo_name: bytes = b"PARAM.SFO;1",
        magic: bytes = b"CD001") -> bytes:
    """A plain image: volume descriptor at 16, root at 20, PSP_GAME at 21, PARAM.SFO at 22."""
    image = bytearray(BLOCK * 24)
    payload = param_sfo or b""
    pvd = bytearray(BLOCK)
    pvd[0] = 1
    pvd[1:6] = magic
    pvd[8:16] = b"PSP GAME"
    root = _record(b"\0", 20, BLOCK, directory=True)
    pvd[156 : 156 + len(root)] = root
    image[16 * BLOCK : 17 * BLOCK] = pvd
    root_dir = _record(b"\0", 20, BLOCK, directory=True) + _record(b"\x01", 20, BLOCK, directory=True)
    root_dir += _record(game_dir, 21, BLOCK, directory=True)
    image[20 * BLOCK : 20 * BLOCK + len(root_dir)] = root_dir
    game = _record(b"\0", 21, BLOCK, directory=True) + _record(b"\x01", 20, BLOCK, directory=True)
    if param_sfo is not None:
        game += _record(b"ICON0.PNG;1", 23, 4, directory=False)
        game += _record(sfo_name, 22, len(payload), directory=False)
    image[21 * BLOCK : 21 * BLOCK + len(game)] = game
    image[22 * BLOCK : 22 * BLOCK + len(payload)] = payload
    return bytes(image)


def read(data: bytes) -> psp_disc.PspDiscFacts:
    return psp_disc.read_facts(io.BytesIO(data))


# --- The pure reads ----------------------------------------------------------


class TestTheDiscId:
    def test_a_plain_iso_states_disc_id(self):
        facts = read(iso(sfo({"CATEGORY": "UG", "DISC_ID": "ULUS10064", "TITLE": "Game"})))
        assert facts == psp_disc.PspDiscFacts(psp_disc.CONTAINER_ISO, "ULUS10064")
        assert psp_disc.game_id(facts) == "ULUS10064"

    def test_the_dash_umd_data_spells_is_dropped(self):
        assert psp_disc.game_id(psp_disc.PspDiscFacts("iso", "ULUS-10064")) == "ULUS10064"

    @pytest.mark.parametrize("disc_id", ["HOMEBREW1", "ulus10064", "ULUS1006", "ULUS100645", "UL-S10064"])
    def test_an_id_of_another_shape_names_nothing(self, disc_id):
        assert psp_disc.game_id(psp_disc.PspDiscFacts("iso", disc_id)) is None

    @pytest.mark.parametrize("magic", [b"CISO\0\0\0\0", b"\0PBP\0\0\1\0", b"MComprHD"])
    def test_a_container_ppsspp_opens_by_magic_is_not_read(self, magic):
        with pytest.raises(psp_disc.PspDiscFormatNotRead):
            read(magic + b"\0" * (BLOCK * 20))

    def test_no_cd001_is_no_disc(self):
        with pytest.raises(psp_disc.PspDiscUnread, match="CD001"):
            read(iso(sfo({"DISC_ID": "ULUS10064"}), magic=b"CD002"))

    def test_no_psp_game_is_no_disc(self):
        with pytest.raises(psp_disc.PspDiscUnread, match="PSP_GAME"):
            read(iso(sfo({"DISC_ID": "ULUS10064"}), game_dir=b"PSX_GAME"))

    def test_names_compare_exactly_once_the_version_is_cut(self):
        assert read(iso(sfo({"DISC_ID": "ULUS10064"}), sfo_name=b"PARAM.SFO")).disc_id == "ULUS10064"
        with pytest.raises(psp_disc.PspDiscUnread, match="PARAM.SFO"):
            read(iso(sfo({"DISC_ID": "ULUS10064"}), sfo_name=b"param.sfo;1"))

    def test_no_param_sfo_or_no_disc_id_is_no_disc(self):
        with pytest.raises(psp_disc.PspDiscUnread):
            read(iso(None))
        with pytest.raises(psp_disc.PspDiscUnread, match="DISC_ID"):
            read(iso(sfo({"TITLE": "Game"})))
        with pytest.raises(psp_disc.PspDiscUnread, match="DISC_ID"):
            read(iso(sfo({"DISC_ID": 7})))

    def test_an_image_cut_short_is_no_disc(self):
        with pytest.raises(psp_disc.PspDiscUnread):
            read(iso(sfo({"DISC_ID": "ULUS10064"}))[: BLOCK * 17])
        with pytest.raises(psp_disc.PspDiscUnread):
            read(b"CD0")


class TestTheParamSfo:
    def test_values_are_kept_by_format(self):
        values = psp_disc.parse_sfo(sfo({"DISC_ID": "ULUS10064", "PARENTAL_LEVEL": 3, "SAVEDATA_PARAMS": b"\0" * 8}))
        assert values == {"DISC_ID": "ULUS10064", "PARENTAL_LEVEL": 3, "SAVEDATA_PARAMS": b"\0" * 8}

    def test_a_savedata_write_carries_its_keys_and_an_install_does_not(self):
        save = psp_disc.parse_sfo_keys(sfo({"CATEGORY": "MS", "SAVEDATA_FILE_LIST": b"\0", "SAVEDATA_PARAMS": b"\0"}))
        install = psp_disc.parse_sfo_keys(sfo({"CATEGORY": "MS", "SAVEDATA_TITLE": "Data"}))
        assert psp_disc.is_savedata(save)
        assert not psp_disc.is_savedata(install)

    @pytest.mark.parametrize(
        "data",
        [
            b"",
            b"PSF\0" + b"\0" * 16,
            struct.pack("<IIIII", 0x46535000, 0x101, 4096, 20, 0),
            struct.pack("<IIIII", 0x46535000, 0x101, 20, 20, 50),
            b"\0" * (psp_disc.MAX_SFO_BYTES + 1),
        ],
        ids=["empty", "wrong-magic", "table-past-the-end", "index-past-the-end", "past-the-bound"],
    )
    def test_what_readsfo_refuses_is_not_an_sfo(self, data):
        with pytest.raises(psp_disc.SfoUnread):
            psp_disc.parse_sfo(data)


# --- The real machine ----------------------------------------------------------


class TestTheRealMachine:
    def test_it_reads_a_disc_and_its_failures(self, tmp_path: Path):
        image = tmp_path / "Game.iso"
        image.write_bytes(iso(sfo({"DISC_ID": "ULUS10064"})))
        cso = tmp_path / "Game.cso"
        cso.write_bytes(b"CISO" + b"\0" * 64)
        junk = tmp_path / "junk.iso"
        junk.write_bytes(b"\0" * (BLOCK * 17))
        machine = RealMachine()
        result = machine.read_psp_disc(str(image))
        assert result.status == PSP_DISC_OK and result.facts is not None
        assert result.facts.disc_id == "ULUS10064"
        assert machine.read_psp_disc(str(cso)).status == PSP_DISC_FORMAT_NOT_READ
        assert machine.read_psp_disc(str(junk)).status == PSP_DISC_NOT_A_DISC
        assert machine.read_psp_disc(str(tmp_path / "none.iso")).status == PSP_DISC_MISSING
        assert machine.read_psp_disc(str(tmp_path)).status == PSP_DISC_MISSING

    def test_it_reads_a_param_sfos_keys_and_its_failures(self, tmp_path: Path):
        good = tmp_path / "PARAM.SFO"
        good.write_bytes(sfo({"CATEGORY": "MS", "SAVEDATA_PARAMS": b"\0"}))
        bad = tmp_path / "BAD.SFO"
        bad.write_bytes(b"not an sfo at all")
        big = tmp_path / "BIG.SFO"
        big.write_bytes(sfo({"CATEGORY": "MS"}) + b"\0" * psp_disc.MAX_SFO_BYTES)
        machine = RealMachine()
        result = machine.read_psp_param_sfo(str(good))
        assert result.status == PSP_PARAM_SFO_OK and result.keys == frozenset({"CATEGORY", "SAVEDATA_PARAMS"})
        assert machine.read_psp_param_sfo(str(bad)).status == PSP_PARAM_SFO_NOT_AN_SFO
        assert machine.read_psp_param_sfo(str(big)).status == PSP_PARAM_SFO_NOT_AN_SFO
        assert machine.read_psp_param_sfo(str(tmp_path / "none")).status == PSP_PARAM_SFO_MISSING


# --- The save unit -------------------------------------------------------------


class TestTheStatement:
    def test_a_prefix_unit_states_members_and_its_prefix(self):
        unit = SaveUnitStatement(SAVE_UNIT_PREFIXED_FOLDERS, ("ULUS10064DATA", "ulus10064cfg"), None, prefix="ULUS10064")
        assert unit.prefix == "ULUS10064"

    @pytest.mark.parametrize(
        ("members", "archive_folder", "prefix", "message"),
        [
            ((), None, None, "prefix is stated exactly"),
            (None, None, "ULUS10064", "members are stated exactly"),
            (("ULES00001DATA",), None, "ULUS10064", "begins with the prefix"),
            ((), "ULUS10064", "ULUS10064", "top level"),
            ((), None, "..", "no entry name"),
        ],
    )
    def test_a_prefix_unit_is_refused_out_of_shape(self, members, archive_folder, prefix, message):
        with pytest.raises(ValueError, match=message):
            SaveUnitStatement(SAVE_UNIT_PREFIXED_FOLDERS, members, archive_folder, prefix=prefix)

    def test_another_kind_carries_no_prefix(self):
        with pytest.raises(ValueError, match="prefix is stated exactly"):
            SaveUnitStatement(atlas.SAVE_UNIT_MEMBERS, (), None, prefix="ULUS10064")

    def test_the_spelling_checks_a_psp_game_id(self):
        assert atlas.checked_save_id(atlas.SAVE_ID_SPELLING_PSP_GAME_ID, "ULUS10064") == "ULUS10064"
        for bad in ("ULUS-10064", "ulus10064", "ULUS10064DATA", ".."):
            with pytest.raises(ValueError):
                atlas.checked_save_id(atlas.SAVE_ID_SPELLING_PSP_GAME_ID, bad)


_VECTORS = Path(__file__).parent.parent / "vectors" / "machines" / "ppsspp-savedata.json"
_BASE: dict[str, Any] = json.loads(_VECTORS.read_text(encoding="utf-8"))["vectors"][1]["input"]
SAVEDATA = "/mnt/sd/retrodeck/saves/PSP/PPSSPP-SA"
ISO = "/mnt/sd/retrodeck/roms/psp/Game.iso"
SAVE_KEYS = ["CATEGORY", "SAVEDATA_FILE_LIST", "SAVEDATA_PARAMS"]


def answer(
    files: dict[str, str],
    sfos: dict[str, Any] | None = None,
    *,
    disc_id: str = "ULUS10064",
    unlistable: tuple[str, ...] = (),
    content_path: str | None = ISO,
) -> atlas.SavefilePlacement:
    inp = copy.deepcopy(_BASE)
    inp["files"].update({ISO: "iso", **files})
    machine = FixtureMachine(
        inp["files"],
        symlinks=inp["symlinks"],
        dirs=inp["dirs"],
        unlistable=unlistable,
        psp_discs={ISO: {"container": "iso", "disc_id": disc_id}},
        psp_param_sfos=sfos or {},
    )
    (install,) = atlas.detect(inp["home"], machine)
    entry = install.emulators_for("psp", content_path=content_path).entries[0]
    placement = entry.savefile_location(content_path=content_path)
    assert isinstance(placement, atlas.SavefilePlacement)
    return placement


def unit_of(placement: atlas.SavefilePlacement) -> SaveUnitStatement | None:
    (group,) = placement.file_set.groups
    return group.unit


class TestTheFoldersAGameOwns:
    def test_members_match_the_prefix_without_case_and_installs_stay_out(self):
        placement = answer(
            {
                f"{SAVEDATA}/ULUS10064DATA00/PARAM.SFO": "s",
                f"{SAVEDATA}/ulus10064cfg/DATA.BIN": "s",
                f"{SAVEDATA}/ULUS10064INSTALL/PARAM.SFO": "s",
                f"{SAVEDATA}/ULUS10065DATA/PARAM.SFO": "s",
            },
            {
                f"{SAVEDATA}/ULUS10064DATA00/PARAM.SFO": SAVE_KEYS,
                f"{SAVEDATA}/ULUS10064INSTALL/PARAM.SFO": ["CATEGORY", "SAVEDATA_TITLE"],
                f"{SAVEDATA}/ULUS10065DATA/PARAM.SFO": SAVE_KEYS,
            },
        )
        assert placement.needs == ()
        assert unit_of(placement) == SaveUnitStatement(
            SAVE_UNIT_PREFIXED_FOLDERS, ("ULUS10064DATA00", "ulus10064cfg"), None, prefix="ULUS10064"
        )

    def test_a_param_sfo_spelled_in_another_case_is_read(self):
        placement = answer(
            {f"{SAVEDATA}/ULUS10064INSTALL/param.sfo": "s"},
            {f"{SAVEDATA}/ULUS10064INSTALL/param.sfo": ["CATEGORY"]},
        )
        unit = unit_of(placement)
        assert unit is not None and unit.members == ()

    @pytest.mark.parametrize(
        ("files", "sfos", "unlistable"),
        [
            ({f"{SAVEDATA}/ULUS10064DATA/PARAM.SFO": "s"}, {f"{SAVEDATA}/ULUS10064DATA/PARAM.SFO": "not-an-sfo"}, ()),
            ({f"{SAVEDATA}/ULUS10064DATA/PARAM.SFO": "s"}, {}, ()),
            ({f"{SAVEDATA}/ULUS10064DATA/PARAM.SFO": "s", f"{SAVEDATA}/ULUS10064DATA/param.sfo": "s"}, {}, ()),
            ({f"{SAVEDATA}/ULUS10064DATA/x": "s"}, {}, (f"{SAVEDATA}/ULUS10064DATA",)),
            ({}, {}, (SAVEDATA,)),
        ],
        ids=["not-an-sfo", "unreadable-sfo", "two-spellings", "folder-unlistable", "savedata-unlistable"],
    )
    def test_folders_not_established_keep_the_hole(self, files, sfos, unlistable):
        placement = answer(files, sfos, unlistable=unlistable)
        assert unit_of(placement) is None
        assert placement.needs == (atlas.HOLE_SAVE_ID,)
        (names,) = [c for c in placement.caveats if c.code == atlas.CAVEAT_FILE_NAMES_UNESTABLISHED]
        assert names.data["save_id_spelling"] == atlas.SAVE_ID_SPELLING_PSP_GAME_ID

    def test_no_content_path_keeps_the_hole(self):
        placement = answer({}, content_path=None)
        assert unit_of(placement) is None and placement.needs == (atlas.HOLE_SAVE_ID,)

    def test_a_disc_id_of_another_shape_keeps_the_hole(self):
        placement = answer({}, disc_id="HOMEBREW1")
        assert unit_of(placement) is None and placement.needs == (atlas.HOLE_SAVE_ID,)

    def test_only_a_psn_id_carries_the_prefix_caveat(self):
        def codes(disc_id: str) -> list[str]:
            return [c.code for c in answer({}, disc_id=disc_id).caveats]

        assert atlas.CAVEAT_PSN_SAVE_PREFIX_MAY_DIFFER in codes("NPJH50804")
        assert atlas.CAVEAT_PSN_SAVE_PREFIX_MAY_DIFFER not in codes("ULJS00445")
