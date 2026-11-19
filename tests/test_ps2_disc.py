"""PCSX2's PS2 save unit: the disc's serial, the folder card's superblock, and the directories a game owns.

Three layers, each proven on its own: the pure reads (:mod:`atlas.ps2_disc`
over synthetic image bytes), the real machine's two reads over files on disk,
and the save answer over fixture machines. The conformance vectors
(``vectors/machines/pcsx2-folder-cards.json``) hold the serialized answers.
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
from atlas import ps2_disc
from atlas.machine import (
    PCSX2_SUPERBLOCK_FORMATTED,
    PCSX2_SUPERBLOCK_MISSING,
    PCSX2_SUPERBLOCK_UNFORMATTED,
    PS2_DISC_FORMAT_NOT_READ,
    PS2_DISC_MISSING,
    PS2_DISC_NOT_A_DISC,
    PS2_DISC_OK,
    FixtureMachine,
    RealMachine,
)
from atlas.placement import SAVE_UNIT_PREFIXED_FOLDERS, SaveUnitStatement

BLOCK = ps2_disc.USER_DATA_SIZE
CNF = b"BOOT2 = cdrom0:\\SLUS_201.52;1\r\nVER = 1.00\r\nVMODE = NTSC\r\n"


# --- Synthetic bytes ---------------------------------------------------------


def _record(name: bytes, lba: int, size: int, *, directory: bool) -> bytes:
    body = struct.pack("<BBI4sI4s7sBBBHH", 0, 0, lba, b"\0" * 4, size, b"\0" * 4, b"\0" * 7,
                       0x02 if directory else 0, 0, 0, 1, 0)
    record = body[:32] + bytes([len(name)]) + name
    if len(record) % 2:
        record += b"\0"
    return bytes([len(record)]) + record[1:]


def user_blocks(system_cnf: bytes | None, *, name: bytes = b"SYSTEM.CNF;1", cnf_directory: bool = False,
                pvd_at: int = 16, magic: bytes = b"CD001") -> list[bytes]:
    """24 blocks of user data: a primary descriptor at *pvd_at*, the root at 20, SYSTEM.CNF at 21."""
    blocks = [bytes(BLOCK)] * 24
    pvd = bytearray(BLOCK)
    pvd[0] = 1
    pvd[1:6] = magic
    root = _record(b"\0", 20, BLOCK, directory=True)
    pvd[156 : 156 + len(root)] = root
    if pvd_at != 16:
        terminator = bytearray(BLOCK)
        terminator[0] = 2  # a supplementary descriptor, skipped
        terminator[1:6] = b"CD001"
        blocks[16] = bytes(terminator)
    blocks[pvd_at] = bytes(pvd)
    entries = _record(b"\0", 20, BLOCK, directory=True) + _record(b"\x01", 20, BLOCK, directory=True)
    entries += _record(b"IOPRP300.IMG;1", 22, 4, directory=False)
    if system_cnf is not None:
        entries += _record(name, 21, len(system_cnf), directory=cnf_directory)
    blocks[20] = entries.ljust(BLOCK, b"\0")
    blocks[21] = (system_cnf or b"").ljust(BLOCK, b"\0")
    return blocks


def image(blocks: list[bytes], *, block_size: int = 2048, offset: int = 0) -> bytes:
    """The user-data blocks laid out the way one of Detect's layouts reads them."""
    user_at = {2048: 0, 2336: 8, 2352: 24, 2448: 24}[block_size]
    out = bytearray(offset)
    for data in blocks:
        raw = bytearray(block_size)
        raw[user_at : user_at + BLOCK] = data
        out += raw
    return bytes(out)


def read(data: bytes, path: str = "Game.iso") -> ps2_disc.Ps2DiscFacts:
    return ps2_disc.read_facts(io.BytesIO(data), len(data), path)


# --- The pure reads ----------------------------------------------------------


class TestTheBootLine:
    @pytest.mark.parametrize(
        ("block_size", "offset"),
        [(2048, 0), (2336, 0), (2352, 0), (2448, 0), (2048, 150 * 2048), (2352, 150 * 2048), (2448, 150 * 2048)],
    )
    def test_every_layout_detect_tries_is_read(self, block_size, offset):
        facts = read(image(user_blocks(CNF), block_size=block_size, offset=offset))
        assert facts == ps2_disc.Ps2DiscFacts("ps2", "cdrom0:\\SLUS_201.52;1")

    def test_the_name_compares_without_case_once_the_version_is_cut(self):
        assert read(image(user_blocks(CNF, name=b"system.cnf;1"))).boot == "cdrom0:\\SLUS_201.52;1"
        assert read(image(user_blocks(CNF, name=b"SYSTEM.CNF"))).boot == "cdrom0:\\SLUS_201.52;1"

    def test_the_primary_descriptor_is_searched_for_past_block_16(self):
        assert read(image(user_blocks(CNF, pvd_at=17))).disc_type == "ps2"

    def test_a_ps1_disc_names_its_boot_line(self):
        facts = read(image(user_blocks(b"BOOT = cdrom:\\SLUS_000.67;1\nTCB = 4\n")))
        assert facts == ps2_disc.Ps2DiscFacts("ps1", "cdrom:\\SLUS_000.67;1")
        assert ps2_disc.product_serial(facts) is None

    @pytest.mark.parametrize(
        "data",
        [
            pytest.param(image(user_blocks(CNF, magic=b"CD002")), id="no-cd001"),
            pytest.param(image(user_blocks(None)), id="no-system-cnf"),
            pytest.param(image(user_blocks(CNF, cnf_directory=True)), id="system-cnf-a-directory"),
            pytest.param(image(user_blocks(b"VMODE = PAL\n")), id="no-boot-line"),
            pytest.param(image(user_blocks(CNF))[: 16 * BLOCK], id="fewer-than-17-blocks"),
        ],
    )
    def test_bytes_that_name_no_game_are_no_disc(self, data):
        with pytest.raises(ps2_disc.Ps2DiscUnread):
            read(data)

    @pytest.mark.parametrize("path", ["Game.chd", "Game.CSO", "Game.zso", "Game.iso.gz", "Game.elf", "Game.cue"])
    def test_a_container_or_another_suffix_is_not_read(self, path):
        with pytest.raises(ps2_disc.Ps2DiscFormatNotRead):
            read(image(user_blocks(CNF)), path)

    def test_the_last_boot_line_wins_and_an_empty_last_value_is_passed_over(self):
        text = b"BOOT = cdrom:\\A.EXE;1\n  BOOT2 =  cdrom0:\\SLES_123.45;1  \nVER =\n"
        assert ps2_disc.boot_line(text) == ("ps2", "cdrom0:\\SLES_123.45;1")
        assert ps2_disc.boot_line(b"BOOT2 = cdrom0:\\X;1\nBOOT2 =\nVER = 1\n") == ("ps2", "")
        assert ps2_disc.boot_line(b"boot2 = cdrom0:\\SLUS_201.52;1\n") is None


class TestTheSerial:
    @pytest.mark.parametrize(
        ("boot", "spelled"),
        [
            ("cdrom0:\\SLUS_201.52;1", "SLUS-20152"),
            ("cdrom0:\\slpm_652.86", "SLPM-65286"),
            ("cdrom0:SCES_503.60;1", "SCES-50360"),
            ("cdrom0:\\DIR\\SLES-508.21;1", "SLES-50821"),
            ("SLUS_201.52", "SLUS-20152"),
            ("cdrom0:\\GAME.ELF;1", ""),
            ("", ""),
        ],
    )
    def test_executable_path_to_serial(self, boot, spelled):
        assert ps2_disc.serial(ps2_disc.Ps2DiscFacts("ps2", boot)) == spelled

    def test_only_a_product_shaped_serial_names_a_unit(self):
        assert ps2_disc.product_serial(ps2_disc.Ps2DiscFacts("ps2", "cdrom0:\\SLUS_201.52;1")) == "SLUS-20152"
        assert ps2_disc.product_serial(ps2_disc.Ps2DiscFacts("ps2", "cdrom0:\\SLUS_201.52X;1")) is None

    def test_the_gamedb_filters_are_found_without_case(self):
        assert ps2_disc.memcard_filters("SLES-82012") == ("SLES-82011",)
        assert ps2_disc.memcard_filters("sles-82012") == ("SLES-82011",)
        assert ps2_disc.memcard_filters("SLUS-20152") is None


# --- The real machine ---------------------------------------------------------


class TestTheRealMachine:
    def test_it_reads_a_disc_and_its_failures(self, tmp_path: Path):
        good = tmp_path / "Game.iso"
        good.write_bytes(image(user_blocks(CNF)))
        raw = tmp_path / "Game.bin"
        raw.write_bytes(image(user_blocks(CNF), block_size=2352))
        empty = tmp_path / "Empty.iso"
        empty.write_bytes(image(user_blocks(None)))
        chd = tmp_path / "Game.chd"
        chd.write_bytes(b"MComprHD")
        machine = RealMachine()
        assert machine.read_ps2_disc(str(good)).status == PS2_DISC_OK
        assert machine.read_ps2_disc(str(raw)).facts == ps2_disc.Ps2DiscFacts("ps2", "cdrom0:\\SLUS_201.52;1")
        assert machine.read_ps2_disc(str(empty)).status == PS2_DISC_NOT_A_DISC
        assert machine.read_ps2_disc(str(chd)).status == PS2_DISC_FORMAT_NOT_READ
        assert machine.read_ps2_disc(str(tmp_path / "Gone.iso")).status == PS2_DISC_MISSING
        assert machine.read_ps2_disc(str(tmp_path)).status == PS2_DISC_FORMAT_NOT_READ

    def test_it_reads_a_superblock_the_way_is_formatted_does(self, tmp_path: Path):
        whole = bytearray(0x2000)
        whole[0x16] = 0x6F
        (tmp_path / "good").write_bytes(bytes(whole))
        (tmp_path / "short").write_bytes(bytes(whole[:0x200]))
        (tmp_path / "other").write_bytes(bytes(0x2000))
        (tmp_path / "dir").mkdir()
        machine = RealMachine()
        assert machine.read_pcsx2_superblock(str(tmp_path / "good")) == PCSX2_SUPERBLOCK_FORMATTED
        assert machine.read_pcsx2_superblock(str(tmp_path / "short")) == PCSX2_SUPERBLOCK_UNFORMATTED
        assert machine.read_pcsx2_superblock(str(tmp_path / "other")) == PCSX2_SUPERBLOCK_UNFORMATTED
        assert machine.read_pcsx2_superblock(str(tmp_path / "dir")) == PCSX2_SUPERBLOCK_MISSING
        assert machine.read_pcsx2_superblock(str(tmp_path / "gone")) == PCSX2_SUPERBLOCK_MISSING

    def test_a_real_superblock_starts_with_the_card_format_string(self):
        # Byte 0x16 of "Sony PS2 Memory Card Format " is its 'o' — 0x6F.
        assert b"Sony PS2 Memory Card Format "[0x16] == 0x6F


# --- The save answer ----------------------------------------------------------

_VECTORS = Path(__file__).parent.parent / "vectors" / "machines" / "pcsx2-folder-cards.json"
_BASE: dict[str, Any] = json.loads(_VECTORS.read_text(encoding="utf-8"))["vectors"][0]["input"]
CARD = "/mnt/sd/retrodeck/saves/ps2/pcsx2/memcards/Mcd001.ps2"
ISO = "/mnt/sd/retrodeck/roms/ps2/Game.iso"


def answer(
    card_files: tuple[str, ...],
    *,
    boot: str = "cdrom0:\\SLUS_201.52;1",
    disc_type: str = "ps2",
    unlistable: tuple[str, ...] = (),
    inaccessible: tuple[str, ...] = (),
    content_path: str | None = ISO,
) -> atlas.SavefilePlacement:
    inp = copy.deepcopy(_BASE)
    files = {k: v for k, v in inp["files"].items() if not k.startswith(f"{CARD}/") or k.endswith("_pcsx2_superblock")}
    files.update({f"{CARD}/{name}": "save" for name in card_files})
    machine = FixtureMachine(
        files,
        dirs=inp["dirs"],
        unlistable=unlistable,
        inaccessible=inaccessible,
        ps2_discs={ISO: {"disc_type": disc_type, "boot": boot}},
        pcsx2_superblocks=inp["pcsx2_superblocks"],
    )
    (install,) = atlas.detect(inp["home"], machine)
    entry = install.emulators_for("ps2", content_path=content_path).entries[0]
    placement = entry.savefile_location(content_path=content_path)
    assert isinstance(placement, atlas.SavefilePlacement)
    return placement


def unit_of(placement: atlas.SavefilePlacement) -> SaveUnitStatement | None:
    return placement.file_set.groups[0].unit


def codes(placement: atlas.SavefilePlacement) -> list[str]:
    return [caveat.code for caveat in placement.caveats]


class TestTheDirectoriesAGameOwns:
    def test_members_begin_with_the_prefix_and_the_cards_own_files_stay_out(self):
        placement = answer(
            ("BASLUS-20152AC04/a", "BASLUS-20152SYS/a", "_pcsx2_deleted_BASLUS-20152X/a", "BASLUS-20152.txt",
             "BESLES-50000/a")
        )
        assert placement.needs == ()
        assert unit_of(placement) == SaveUnitStatement(
            SAVE_UNIT_PREFIXED_FOLDERS, ("BASLUS-20152AC04", "BASLUS-20152SYS"), None, prefix="BASLUS-20152"
        )
        assert atlas.CAVEAT_PS2_SAVE_PREFIX_MAY_DIFFER in codes(placement)

    def test_a_directory_in_another_case_pcsx2_does_not_show_states_no_unit(self):
        # FilterMatches compares exactly, the unit's match without ASCII case.
        placement = answer(("BASLUS-20152AC04/a", "BAslus-20152OLD/a"))
        assert unit_of(placement) is None
        assert atlas.CAVEAT_PS2_SAVE_PREFIX_MAY_DIFFER not in codes(placement)

    def test_a_directory_containing_the_serial_elsewhere_states_no_unit(self):
        assert unit_of(answer(("BASLUS-20152AC04/a", "XSLUS-20152/a"))) is None

    def test_a_never_saved_game_of_an_unknown_family_states_no_unit(self):
        assert unit_of(answer(("BASLUS-20152AC04/a",), boot="cdrom0:\\SCAJ_200.03;1")) is None

    def test_a_never_saved_game_of_a_known_family_takes_its_letter(self):
        unit = unit_of(answer((), boot="cdrom0:\\SCPS_150.10;1"))
        assert unit is not None and unit.prefix == "BISCPS-15010" and unit.members == ()

    def test_a_gamedb_list_of_several_states_no_unit(self):
        # SCPS-15009's entry lists PBPX-95502 and SCPS-15009: either may hold its save.
        assert unit_of(answer(("BISCPS-15009GT3/a",), boot="cdrom0:\\SCPS_150.09;1")) is None

    def test_a_ps1_disc_states_no_unit(self):
        assert unit_of(answer(("BASLUS-20152AC04/a",), disc_type="ps1", boot="cdrom:\\SLUS_000.67;1")) is None

    def test_a_card_that_cannot_be_listed_or_looked_into_states_no_unit(self):
        assert unit_of(answer(("BASLUS-20152AC04/a",), unlistable=(CARD,))) is None
        assert unit_of(answer(("BASLUS-20152AC04/a",), inaccessible=(f"{CARD}/BASLUS-20152AC04",))) is None

    def test_no_content_path_leaves_the_tree(self):
        placement = answer(("BASLUS-20152AC04/a",), content_path=None)
        assert unit_of(placement) is None
        assert atlas.CAVEAT_PS2_SAVE_PREFIX_MAY_DIFFER not in codes(placement)
