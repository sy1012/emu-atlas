"""RPCS3's PS3 save unit: the title id a game's file states, and the savedata folders it begins.

Three layers, each proven on its own: the pure read (:mod:`atlas.ps3_title`
over synthetic PARAM.SFO, package and ISO bytes), the real machine's read over
files on disk, and the save answer over fixture machines. The conformance
vectors (``vectors/machines/rpcs3-savedata.json``) hold the serialized
answers.
"""

from __future__ import annotations

import copy
import io
import json
import struct
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

import atlas
from atlas import ps3_file, ps3_title
from atlas.machine import (
    PS3_TITLE_EXECUTABLE,
    PS3_TITLE_FORMAT_NOT_READ,
    PS3_TITLE_MISSING,
    PS3_TITLE_NO_TITLE_ID,
    PS3_TITLE_OK,
    FixtureMachine,
    RealMachine,
)
from atlas.placement import SAVE_UNIT_PREFIXED_FOLDERS, SaveUnitStatement

SECTOR = 2048


# --- Synthetic bytes ---------------------------------------------------------


def sfo(entries: Mapping[str, str | int]) -> bytes:
    """A PARAM.SFO ``psf::load`` keeps: strings as 0x0204 padded to four bytes, ints as 0x0404."""
    keys = b""
    data = b""
    index = b""
    for key, value in entries.items():
        if isinstance(value, str):
            raw = value.encode() + b"\0"
            fmt, length, maximum = 0x0204, len(raw), len(raw) + 3 & ~3
            raw = raw.ljust(maximum, b"\0")
        else:
            raw, fmt, length, maximum = struct.pack("<I", value), 0x0404, 4, 4
        index += struct.pack("<HHIII", len(keys), fmt, length, maximum, len(data))
        keys += key.encode() + b"\0"
        data += raw
    key_start = 20 + len(index)
    keys = keys.ljust(len(keys) + 3 & ~3, b"\0")
    data_start = key_start + len(keys)
    header = struct.pack("<IIIII", 0x46535000, 0x101, key_start, data_start, len(entries))
    # A trailing pad, so the loader's check that a value's maximum lies short of the file's end holds.
    return header + index + keys + data + b"\0" * 8


GAME_SFO = {"CATEGORY": "DG", "TITLE": "Game", "TITLE_ID": "BLUS30443"}


def pkg(content_id: bytes = b"UP0006-NPUB30058_00-3ON3NHLARCADE001", *, pkg_type: int = 0x8000,
        platform: int = 0x0001, magic: bytes = b"\x7fPKG") -> bytes:
    """A package header: magic, big-endian type and platform, the content id at 0x30."""
    head = bytearray(0xC0)
    head[0:4] = magic
    struct.pack_into(">HH", head, 4, pkg_type, platform)
    head[0x30 : 0x30 + len(content_id)] = content_id
    return bytes(head)


def _record(name: bytes, lba: int, size: int, *, directory: bool) -> bytes:
    body = bytearray(33)
    struct.pack_into("<I", body, 2, lba)
    struct.pack_into("<I", body, 10, size)
    body[25] = 0x02 if directory else 0
    body[32] = len(name)
    record = bytes(body) + name
    if len(record) % 2:
        record += b"\0"
    return bytes([len(record)]) + record[1:]


def iso(param_sfo: bytes | None, *, game_dir: bytes = b"PS3_GAME") -> bytes:
    """An image: a primary descriptor at 16, the terminator at 17, root at 20, PS3_GAME at 21, PARAM.SFO at 22."""
    image = bytearray(SECTOR * 26)
    struct.pack_into(">I", image, 0, 1)
    pvd = bytearray(SECTOR)
    pvd[0] = 1
    pvd[1:6] = b"CD001"
    root = _record(b"\0", 20, SECTOR, directory=True)
    pvd[156 : 156 + len(root)] = root
    image[16 * SECTOR : 17 * SECTOR] = pvd
    image[17 * SECTOR] = 255
    image[17 * SECTOR + 1 : 17 * SECTOR + 6] = b"CD001"
    root_dir = _record(b"\0", 20, SECTOR, directory=True) + _record(b"\x01", 20, SECTOR, directory=True)
    root_dir += _record(game_dir, 21, SECTOR, directory=True)
    image[20 * SECTOR : 20 * SECTOR + len(root_dir)] = root_dir
    game = _record(b"\0", 21, SECTOR, directory=True) + _record(b"\x01", 20, SECTOR, directory=True)
    payload = param_sfo or b""
    if param_sfo is not None:
        game += _record(b"PARAM.SFO;1", 22, len(payload), directory=False)
    image[21 * SECTOR : 21 * SECTOR + len(game)] = game
    image[22 * SECTOR : 22 * SECTOR + len(payload)] = payload
    return bytes(image)


def read(data: bytes) -> ps3_title.Ps3TitleFacts:
    return ps3_title.read_facts(io.BytesIO(data), len(data))


# --- The pure read -------------------------------------------------------------


class TestTheTitleId:
    def test_a_param_sfo_states_title_id(self):
        assert read(sfo(GAME_SFO)) == ps3_title.Ps3TitleFacts(ps3_title.SOURCE_PARAM_SFO, "BLUS30443")

    def test_a_package_names_the_title_inside_its_content_id(self):
        facts = read(pkg())
        assert facts == ps3_title.Ps3TitleFacts(ps3_title.SOURCE_PKG, "NPUB30058", content_type=0)
        assert ps3_title.title_id(facts) == "NPUB30058"

    def test_a_debug_package_is_read_too(self):
        assert read(pkg(pkg_type=0x0000)).title_id == "NPUB30058"

    def test_an_image_states_its_ps3_game_param_sfo(self):
        assert read(iso(sfo(GAME_SFO))) == ps3_title.Ps3TitleFacts(ps3_title.SOURCE_ISO, "BLUS30443")

    @pytest.mark.parametrize("magic", [b"SCE\0", b"\x7fELF"])
    def test_an_executable_is_told_apart(self, magic):
        with pytest.raises(ps3_title.Ps3TitleExecutable):
            read(magic + b"\0" * 64)

    def test_a_psp_package_names_no_ps3_game(self):
        with pytest.raises(ps3_title.Ps3TitleFormatNotRead):
            read(pkg(platform=0x0002))

    def test_anything_else_is_not_read(self):
        with pytest.raises(ps3_title.Ps3TitleFormatNotRead):
            read(b"\0" * (SECTOR * 18))
        with pytest.raises(ps3_title.Ps3TitleFormatNotRead):
            read(b"")

    @pytest.mark.parametrize(
        "data",
        [
            pkg(pkg_type=0x1234),
            pkg(platform=0x0003),
            pkg(content_id=b""),
            pkg()[:0x40],
        ],
        ids=["unknown-type", "unknown-platform", "no-content-id", "short-header"],
    )
    def test_a_package_rpcs3_refuses_names_no_title(self, data):
        with pytest.raises(ps3_title.Ps3TitleUnread):
            read(data)

    @pytest.mark.parametrize(
        "entries",
        [
            {"CATEGORY": "DG", "TITLE": "Game"},
            {"CATEGORY": "XX", "TITLE_ID": "BLUS30443"},
            {"TITLE_ID": "BLUS30443"},
            {"CATEGORY": "DG", "TITLE_ID": 7},
        ],
        ids=["no-title-id", "category-the-loader-clears", "no-category", "an-integer"],
    )
    def test_a_param_sfo_naming_no_title_is_unread(self, entries):
        with pytest.raises(ps3_title.Ps3TitleUnread):
            read(sfo(entries))

    def test_a_param_sfo_past_the_bound_is_unread(self):
        with pytest.raises(ps3_title.Ps3TitleUnread):
            read(sfo(GAME_SFO) + b"\0" * ps3_file.PSF_SIZE_LIMIT)

    def test_an_image_without_ps3_game_names_no_title(self):
        with pytest.raises(ps3_title.Ps3TitleUnread):
            read(iso(sfo(GAME_SFO), game_dir=b"PS3_GAMX"))
        with pytest.raises(ps3_title.Ps3TitleUnread):
            read(iso(None))

    def test_an_encrypted_param_sfo_in_an_image_names_no_title(self):
        with pytest.raises(ps3_title.Ps3TitleUnread):
            read(iso(b"\x13\x37" * 64))

    @pytest.mark.parametrize("spelled", ["blus30443", "BLUS-30443", "BLUS3044", "HOMEBREW1"])
    def test_an_id_of_another_shape_names_nothing(self, spelled):
        assert ps3_title.title_id(ps3_title.Ps3TitleFacts(ps3_title.SOURCE_PARAM_SFO, spelled)) is None

    def test_the_spelling_checks_a_ps3_title_id(self):
        assert atlas.checked_save_id(atlas.SAVE_ID_SPELLING_PS3_TITLE_ID, "BLUS30443") == "BLUS30443"
        for bad in ("BLUS-30443", "blus30443", "BLUS30443GAMEDATA", ".."):
            with pytest.raises(ValueError):
                atlas.checked_save_id(atlas.SAVE_ID_SPELLING_PS3_TITLE_ID, bad)


# --- The real machine ----------------------------------------------------------


class TestTheRealMachine:
    def test_it_reads_each_form_and_its_failures(self, tmp_path: Path):
        files = {
            "PARAM.SFO": sfo(GAME_SFO),
            "Game.pkg": pkg(),
            "Game.iso": iso(sfo(GAME_SFO)),
            "EBOOT.BIN": b"SCE\0" + b"\0" * 64,
            "junk.bin": b"junk",
            "BAD.SFO": sfo({"CATEGORY": "DG"}),
        }
        for name, data in files.items():
            (tmp_path / name).write_bytes(data)
        machine = RealMachine()

        def status(name: str) -> str:
            return machine.read_ps3_title(str(tmp_path / name)).status

        result = machine.read_ps3_title(str(tmp_path / "Game.pkg"))
        assert result.status == PS3_TITLE_OK and result.facts == ps3_title.Ps3TitleFacts("pkg", "NPUB30058", 0)
        assert status("PARAM.SFO") == PS3_TITLE_OK and status("Game.iso") == PS3_TITLE_OK
        assert status("EBOOT.BIN") == PS3_TITLE_EXECUTABLE
        assert status("junk.bin") == PS3_TITLE_FORMAT_NOT_READ
        assert status("BAD.SFO") == PS3_TITLE_NO_TITLE_ID
        assert status("none") == PS3_TITLE_MISSING
        assert machine.read_ps3_title(str(tmp_path)).status == PS3_TITLE_MISSING


# --- The save unit -------------------------------------------------------------

_VECTORS = Path(__file__).parent.parent / "vectors" / "machines" / "rpcs3-savedata.json"
_BASE: dict[str, Any] = json.loads(_VECTORS.read_text(encoding="utf-8"))["vectors"][4]["input"]
HDD = "/mnt/sd/hdd0"
SAVEDATA = f"{HDD}/home/00000001/savedata"
DUMP = "/mnt/sd/retrodeck/roms/ps3/Game.ps3"
SFO = f"{DUMP}/PS3_GAME/PARAM.SFO"


def answer(
    files: dict[str, str] | None = None,
    titles: dict[str, Any] | None = None,
    *,
    unlistable: tuple[str, ...] = (),
    inaccessible: tuple[str, ...] = (),
    content_path: str | None = DUMP,
    drop: tuple[str, ...] = (),
) -> atlas.SavefilePlacement:
    inp = copy.deepcopy(_BASE)
    for path in drop:
        del inp["files"][path]
    inp["files"].update(files or {})
    machine = FixtureMachine(
        inp["files"],
        dirs=inp["dirs"],
        unlistable=unlistable,
        inaccessible=inaccessible,
        ps3_titles={SFO: {"source": "param-sfo", "title_id": "BLUS30443"}} if titles is None else titles,
    )
    (install,) = atlas.detect(inp["home"], machine)
    entry = install.emulators_for("ps3", content_path=content_path).entries[0]
    placement = entry.savefile_location(content_path=content_path)
    assert isinstance(placement, atlas.SavefilePlacement)
    return placement


def unit_of(placement: atlas.SavefilePlacement) -> SaveUnitStatement | None:
    return placement.file_set.groups[0].unit


def keeps_the_hole(placement: atlas.SavefilePlacement) -> bool:
    (names,) = [c for c in placement.caveats if c.code == atlas.CAVEAT_FILE_NAMES_UNESTABLISHED]
    return (
        all(group.unit is None for group in placement.file_set.groups)
        and placement.needs == (atlas.HOLE_SAVE_ID,)
        and names.data["save_id_spelling"] == atlas.SAVE_ID_SPELLING_PS3_TITLE_ID
        and atlas.CAVEAT_PS3_SAVE_PREFIX_MAY_DIFFER not in [c.code for c in placement.caveats]
    )


class TestTheFoldersAGameOwns:
    def test_members_match_the_prefix_without_case(self):
        placement = answer(
            {
                f"{SAVEDATA}/BLUS30443GAMEDATA/PARAM.SFO": "s",
                f"{SAVEDATA}/blus30443cfg/DATA.BIN": "s",
                f"{SAVEDATA}/BLUS30444DATA/PARAM.SFO": "s",
                f"{SAVEDATA}/.working_BLUS30443GAMEDATA/PARAM.SFO": "s",
            }
        )
        assert placement.needs == ()
        assert unit_of(placement) == SaveUnitStatement(
            SAVE_UNIT_PREFIXED_FOLDERS, ("BLUS30443GAMEDATA", "blus30443cfg"), None, prefix="BLUS30443"
        )
        (prefix,) = [c for c in placement.caveats if c.code == atlas.CAVEAT_PS3_SAVE_PREFIX_MAY_DIFFER]
        assert prefix.data["prefix"] == "BLUS30443" and prefix.data["dir"] == SAVEDATA

    def test_a_dump_without_its_disc_sfb_reads_the_same_param_sfo(self):
        placement = answer(drop=(f"{DUMP}/PS3_DISC.SFB",))
        unit = unit_of(placement)
        assert unit is not None and unit.prefix == "BLUS30443"

    def test_an_installed_style_folder_reads_its_own_param_sfo(self):
        folder = "/mnt/sd/retrodeck/roms/ps3/NPUB30058"
        placement = answer(
            {f"{folder}/USRDIR/EBOOT.BIN": "self", f"{folder}/PARAM.SFO": "sfo"},
            {f"{folder}/PARAM.SFO": {"source": "param-sfo", "title_id": "NPUB30058"}},
            content_path=folder,
        )
        unit = unit_of(placement)
        assert unit is not None and unit.prefix == "NPUB30058"

    @pytest.mark.parametrize(
        ("titles", "content_path", "inaccessible"),
        [
            ({}, None, ()),
            ({SFO: "no-title-id"}, DUMP, ()),
            ({SFO: {"source": "param-sfo", "title_id": "HOMEBREW1"}}, DUMP, ()),
            ({SFO: {"source": "pkg", "title_id": "BLUS30443"}}, DUMP, ()),
            ({}, DUMP, ()),
            ({}, "/mnt/sd/retrodeck/roms/ps3/Empty.ps3", ()),
            ({}, DUMP, (f"{DUMP}/PS3_GAME/PS3_DISC.SFB",)),
            ({}, DUMP, (f"{DUMP}/USRDIR/EBOOT.BIN",)),
        ],
        ids=["no-content-path", "no-title-id", "another-shape", "not-a-param-sfo", "sfo-unmodelled",
             "no-boot-file", "sfb-unlookable", "boot-file-unlookable"],
    )
    def test_a_title_not_read_keeps_the_hole(self, titles, content_path, inaccessible):
        files = {"/mnt/sd/retrodeck/roms/ps3/Empty.ps3/README": "x"}
        assert keeps_the_hole(answer(files, titles, content_path=content_path, inaccessible=inaccessible))

    def test_two_accounts_keep_the_hole(self):
        placement = answer({f"{HDD}/home/00000002/localusername": "User"})
        assert len(placement.file_set.groups) == 3
        assert keeps_the_hole(placement)

    def test_an_undecided_account_beside_the_one_keeps_the_hole(self):
        placement = answer({f"{HDD}/home/00000002/x": "x"}, inaccessible=(f"{HDD}/home/00000002/localusername",))
        assert keeps_the_hole(placement)

    def test_a_matching_entry_of_unknown_kind_keeps_the_hole(self):
        assert keeps_the_hole(answer(inaccessible=(f"{SAVEDATA}/BLUS30443GAMEDATA",)))

    def test_an_unlistable_savedata_keeps_the_hole(self):
        assert keeps_the_hole(answer(unlistable=(SAVEDATA,)))
