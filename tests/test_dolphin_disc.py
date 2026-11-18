"""Reading a GameCube or Wii disc image the way Dolphin reads it, and the save answers that name a game from it.

The reader (:mod:`atlas.dolphin_disc`) is driven with tiny synthetic images
built in :mod:`tests.dolphin_disc_writer`, one per container, and the reads
that must refuse — a truncated file, a wrong magic, a container whose header
does not hold. The answers are driven through a RetroDECK fixture machine
whose ``dolphin_discs`` declare what the read yields, for the standalone
Dolphin and for the libretro core, and the save-unit statement is held to the
two Dolphin answers alone.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

import atlas
from atlas import dolphin_disc
from atlas.contract import savefile_placement_contract
from atlas.machine import (
    DOLPHIN_DISC_MISSING,
    DOLPHIN_DISC_NOT_A_DISC,
    DOLPHIN_DISC_OK,
    DOLPHIN_DISC_UNREADABLE,
    FixtureMachine,
    RealMachine,
)
from atlas.placement import SAVE_UNIT_FOLDER, SAVE_UNIT_MEMBERS, SavefilePlacement, SaveUnitStatement
from tests import dolphin_disc_writer as w
from tests.answers import placed


def _read(image: bytes) -> dolphin_disc.DolphinDiscFacts:
    return dolphin_disc.read_facts(io.BytesIO(image))


GC = w.gc_disc(b"GALE01", region=1)
WII = w.wii_disc(b"RSBE01", title_id=0x0001000052534245)


class TestTheContainers:
    @pytest.mark.parametrize(
        ("image", "container"),
        [
            (GC, "plain"),
            (w.ciso(GC), "ciso"),
            (w.gcz(GC), "gcz"),
            (w.gcz(GC, raw=(0,)), "gcz"),
        ],
        ids=["plain", "ciso", "gcz", "gcz-raw-block"],
    )
    def test_a_gamecube_disc_reads_its_id_and_region_code_through_every_container_exposing_0x458(
        self, image, container
    ):
        facts = _read(image)
        assert (facts.container, facts.platform, facts.game_id, facts.region_code) == (
            container,
            "gc",
            "GALE01",
            1,
        )
        assert facts.title_id is None

    @pytest.mark.parametrize("rvz", [False, True], ids=["wia", "rvz"])
    def test_a_wia_or_rvz_image_exposes_the_disc_header_alone(self, rvz):
        facts = _read(w.wia(GC, rvz=rvz))
        assert (facts.container, facts.game_id, facts.region_code) == ("rvz" if rvz else "wia", "GALE01", None)
        wii = _read(w.wia(WII, rvz=rvz))
        assert (wii.platform, wii.game_id, wii.title_id) == ("wii", "RSBE01", None)

    @pytest.mark.parametrize(
        ("image", "container"),
        [(WII, "plain"), (w.ciso(WII), "ciso"), (w.gcz(WII), "gcz"), (w.wbfs(WII), "wbfs")],
        ids=["plain", "ciso", "gcz", "wbfs"],
    )
    def test_a_wii_disc_reads_its_game_partitions_ticket_through_every_container_exposing_it(self, image, container):
        facts = _read(image)
        assert (facts.container, facts.platform, facts.game_id) == (container, "wii", "RSBE01")
        assert facts.title_id is not None
        assert dolphin_disc.title_words(facts.title_id) == ("00010000", "52534245")

    def test_an_unused_ciso_block_reads_as_zeros_so_no_disc_is_there(self):
        image = bytearray(w.ciso(GC))
        image[8] = 0
        with pytest.raises(dolphin_disc.DolphinDiscUnread):
            _read(bytes(image))


class TestWhatIsNoDisc:
    @pytest.mark.parametrize(
        "image",
        [
            b"",
            GC[:0x10],
            b"\0" * 0x800,
            w.ciso(GC)[:0x100],
            w.gcz(GC)[:0x10],
            w.wia(GC)[:0x40],
            w.wbfs(WII)[:8],
        ],
        ids=["empty", "truncated-header", "no-magic", "short-ciso", "short-gcz", "short-wia", "short-wbfs"],
    )
    def test_a_short_file_or_a_wrong_magic_is_no_disc(self, image):
        with pytest.raises(dolphin_disc.DolphinDiscUnread):
            _read(image)

    def test_a_gcz_whose_data_is_cut_off_is_no_disc(self):
        image = w.gcz(GC)
        with pytest.raises(dolphin_disc.DolphinDiscUnread):
            _read(image[: len(image) - 10])

    def test_a_gcz_block_that_does_not_inflate_is_no_disc(self):
        image = bytearray(w.gcz(GC))
        data = 32 + 12 * 1
        image[data : data + 4] = b"junk"
        with pytest.raises(dolphin_disc.DolphinDiscUnread):
            _read(bytes(image))

    def test_a_wia_whose_header_hash_does_not_hold_is_no_disc(self):
        image = bytearray(w.wia(GC))
        image[0x58] ^= 1
        with pytest.raises(dolphin_disc.DolphinDiscUnread):
            _read(bytes(image))

    def test_a_wia_of_a_version_dolphin_does_not_read_is_no_disc(self):
        with pytest.raises(dolphin_disc.DolphinDiscUnread):
            _read(w.wia(GC, version=0x02000000))

    def test_a_wbfs_with_no_disc_in_its_first_slot_is_no_disc(self):
        image = bytearray(w.wbfs(WII))
        image[12] = 0
        with pytest.raises(dolphin_disc.DolphinDiscUnread):
            _read(bytes(image))


class TestTheWiiTicket:
    def test_a_ticket_whose_signature_type_es_does_not_know_names_no_title(self):
        assert _read(w.wii_disc(signature=0x12345678)).title_id is None

    def test_a_disc_with_no_game_partition_names_no_title(self):
        assert _read(w.wii_disc(partition_type=1)).title_id is None

    def test_the_game_partition_is_found_in_a_later_group(self):
        assert _read(w.wii_disc(groups=(2,))).title_id == 0x0001000052534245

    def test_the_low_word_of_a_disc_without_its_ticket_is_its_game_code(self):
        assert dolphin_disc.game_code_word(_read(w.wia(WII))) == "52534245"


class TestTheGameCubeRegion:
    @pytest.mark.parametrize(
        ("code", "region"),
        [(0, dolphin_disc.REGION_NTSC_J), (1, dolphin_disc.REGION_NTSC_U), (2, dolphin_disc.REGION_PAL),
         (4, dolphin_disc.REGION_NTSC_J)],
    )  # fmt: skip
    def test_the_region_code_files_the_card(self, code, region):
        named = dolphin_disc.gamecube_region(_read(w.gc_disc(region=code)))
        assert named == dolphin_disc.GameCubeRegion(region, "region-code")

    @pytest.mark.parametrize("code", [3, 5, 0xFFFF])
    def test_an_unknown_region_code_establishes_nothing(self, code):
        # Unknown is replaced by the configured fallback region, which the disc does not state.
        assert dolphin_disc.gamecube_region(_read(w.gc_disc(region=code))) is None

    @pytest.mark.parametrize(
        ("game_id", "revision", "region"),
        [
            (b"GALE01", 0, dolphin_disc.REGION_NTSC_U),
            (b"GALE01", 0x30, dolphin_disc.REGION_NTSC_J),
            (b"GALP01", 0, dolphin_disc.REGION_PAL),
            (b"GALJ01", 0, dolphin_disc.REGION_NTSC_J),
            (b"GALK01", 0, dolphin_disc.REGION_NTSC_J),
            (b"GALX01", 0, None),
            (b"GAL\x0201", 0, None),
        ],
    )
    def test_without_the_region_code_the_country_code_stands_in_where_it_decides_alone(
        self, game_id, revision, region
    ):
        named = dolphin_disc.gamecube_region(_read(w.wia(w.gc_disc(game_id, revision=revision))))
        assert named == (None if region is None else dolphin_disc.GameCubeRegion(region, "country-code"))

    def test_the_game_id_is_filtered_the_way_dolphin_filters_it(self):
        facts = _read(w.gc_disc(b"GA\x00E0!"))
        assert facts.dolphin_game_id == "GA-E0-"
        assert facts.game_code == "GA-E"


class TestTheMachineSeam:
    def test_the_real_machine_reads_an_image_and_states_every_failure(self, tmp_path: Path):
        image = tmp_path / "Melee.iso"
        image.write_bytes(GC)
        junk = tmp_path / "junk.iso"
        junk.write_bytes(b"\0" * 0x40)
        machine = RealMachine()
        read = machine.read_dolphin_disc(str(image))
        assert read.status == DOLPHIN_DISC_OK
        assert read.facts is not None and read.facts.game_id == "GALE01"
        assert machine.read_dolphin_disc(str(junk)).status == DOLPHIN_DISC_NOT_A_DISC
        assert machine.read_dolphin_disc(str(tmp_path / "gone.iso")).status == DOLPHIN_DISC_MISSING
        assert machine.read_dolphin_disc(str(tmp_path)).status == DOLPHIN_DISC_MISSING

    def test_the_fixture_answers_what_it_declares_and_unreadable_for_a_file_it_does_not(self):
        facts = {"container": "rvz", "platform": "gc", "game_id": "GALE01", "revision": 0,
                 "region_code": None, "title_id": None}  # fmt: skip
        machine = FixtureMachine(
            {"/roms/a.rvz": {"size": 1}, "/roms/b.rvz": {"size": 1}, "/roms/c.rvz": {"size": 1}},
            dolphin_discs={"/roms/a.rvz": facts, "/roms/b.rvz": "not-a-disc"},
        )
        assert machine.read_dolphin_disc("/roms/a.rvz").status == DOLPHIN_DISC_OK
        assert machine.read_dolphin_disc("/roms/b.rvz").status == DOLPHIN_DISC_NOT_A_DISC
        assert machine.read_dolphin_disc("/roms/c.rvz").status == DOLPHIN_DISC_UNREADABLE
        assert machine.read_dolphin_disc("/roms/d.rvz").status == DOLPHIN_DISC_MISSING

    @pytest.mark.parametrize(
        "spec",
        [
            {"container": "rvz", "platform": "gc", "game_id": "GALE01", "revision": 0, "region_code": None},
            {"container": "nfs", "platform": "gc", "game_id": "GALE01", "revision": 0, "region_code": None,
             "title_id": None},
            {"container": "rvz", "platform": "gc", "game_id": "GALE", "revision": 0, "region_code": None,
             "title_id": None},
            {"container": "rvz", "platform": "gc", "game_id": "GALE01", "revision": 0, "region_code": None,
             "title_id": "0001000047414c45"},
            {"container": "rvz", "platform": "wii", "game_id": "GALE01", "revision": 0, "region_code": None,
             "title_id": "0001000047414C45"},
        ],
        ids=["missing-field", "unknown-container", "short-id", "title-on-gamecube", "uppercase-title"],
    )  # fmt: skip
    def test_the_fixture_refuses_facts_no_read_yields(self, spec):
        with pytest.raises(ValueError, match="dolphin disc"):
            FixtureMachine({"/roms/a.rvz": {"size": 1}}, dolphin_discs={"/roms/a.rvz": spec})


class TestTheStatement:
    def test_a_folder_names_no_members_and_members_name_their_files(self):
        assert SaveUnitStatement(SAVE_UNIT_FOLDER, None, "52534245").members is None
        assert SaveUnitStatement(SAVE_UNIT_MEMBERS, (), None).members == ()

    @pytest.mark.parametrize(
        ("kind", "members", "archive"),
        [
            ("own-folder", ("a.gci",), None),
            ("own-files", None, None),
            ("own-files", ("../a.gci",), None),
            ("own-files", ("a.gci", "a.gci"), None),
            ("own-folder", None, ".."),
            ("disc", None, None),
        ],
    )
    def test_a_statement_that_does_not_hold_together_is_refused(self, kind, members, archive):
        with pytest.raises(ValueError, match="SaveUnitStatement"):
            SaveUnitStatement(kind, members, archive)


HOME = "/home/deck"
RD = f"{HOME}/.var/app/net.retrodeck.retrodeck"
ES = (
    "/var/lib/flatpak/app/net.retrodeck.retrodeck/current/active/files/retrodeck/components/es-de"
    "/share/es-de/resources/systems/linux/es_systems.xml"
)
CORE = "/var/lib/flatpak/app/net.retrodeck.retrodeck/current/active/files/cores/dolphin_libretro.so"
SAVES = "/mnt/sd/retrodeck/saves"
STANDALONE_GC = f"{RD}/data/dolphin-emu/GC"
STANDALONE_NAND = f"{RD}/data/dolphin-emu/Wii"
ROM_GC = "/mnt/sd/retrodeck/roms/gc/Melee.rvz"
ROM_WII = "/mnt/sd/retrodeck/roms/wii/Sports.rvz"


def _systems(command: str) -> str:
    return (
        '<?xml version="1.0"?><systemList>'
        + "".join(
            f"<system><name>{name}</name><path>%ROMPATH%/{name}</path><extension>.rvz</extension>"
            f'<command label="Dolphin">{command}</command></system>'
            for name in ("gc", "wii")
        )
        + "</systemList>"
    )


def _disc(platform: str, game_id: str, *, container: str = "rvz", region: int | None = None,
          title: str | None = None) -> dict[str, Any]:  # fmt: skip
    return {"container": container, "platform": platform, "game_id": game_id, "revision": 0,
            "region_code": region, "title_id": title}  # fmt: skip


def _gci(code: bytes, size: int = 8256) -> dict[str, Any]:
    return {"md5:4": hashlib.md5(code).hexdigest(), "size": size}


def _answer(
    system: str,
    *,
    core: bool = False,
    discs: dict[str, Any] | None = None,
    files: dict[str, Any] | None = None,
    dirs: tuple[str, ...] = (),
    unlistable: tuple[str, ...] = (),
    content: str | None = None,
    no_content: bool = False,
) -> SavefilePlacement:
    command = (
        "%EMULATOR_RETROARCH% -L %CORE_RETROARCH%/dolphin_libretro.so %ROM%"
        if core
        else "%EMULATOR_DOLPHIN% -b -e %ROM%"
    )
    machine = FixtureMachine(
        {
            f"{RD}/config/retrodeck/retrodeck.json": (
                '{"paths": {"rd_home_path": "/mnt/sd/retrodeck", "saves_path": "/mnt/sd/retrodeck/saves"}}'
            ),
            f"{RD}/config/retroarch/retroarch.cfg": (
                f'savefile_directory = "{SAVES}"\nsort_savefiles_by_content_enable = "true"\n'
                'sort_savefiles_enable = "false"\nlibretro_directory = "/app/cores"\n'
            ),
            ES: _systems(command),
            ROM_GC: {"size": 100},
            ROM_WII: {"size": 100},
            **(files or {}),
        },
        dirs=[SAVES, f"{SAVES}/{system}", *dirs],
        cores={CORE: {"library_name": "dolphin-emu", "library_version": "fd1aca3a"}},
        unlistable=list(unlistable),
        dolphin_discs=discs or {},
    )
    installs = atlas.detect(HOME, machine)
    entry = installs[0].emulators_for(system).entries[0]
    asked = None if no_content else content or (ROM_GC if system == "gc" else ROM_WII)
    return placed(entry.savefile_location(content_path=asked))


GALE = {ROM_GC: _disc("gc", "GALE01")}
USA_A = f"{STANDALONE_GC}/USA/Card A"


class TestTheStandaloneGameCubeAnswer:
    def test_the_discs_region_folder_and_its_game_codes_files_are_the_answer(self):
        p = _answer(
            "gc",
            discs=GALE,
            files={
                f"{USA_A}/01-GALE-a.gci": _gci(b"GALE"),
                f"{USA_A}/01-GALE-b.GCI": _gci(b"GALE"),
                f"{USA_A}/8P-GZLE-zelda.gci": _gci(b"GZLE"),
                f"{USA_A}/01-GALE-short.gci": {"md5": hashlib.md5(b"GALE").hexdigest(), "size": 4},
                f"{USA_A}/GALE.sav": _gci(b"GALE"),
                f"{STANDALONE_GC}/EUR/Card A/01-GALE-eu.gci": _gci(b"GALE"),
            },
        )
        assert p.dir == USA_A
        assert p.needs == ()
        assert [g.dir for g in p.file_set.groups] == [USA_A]
        assert p.file_set.groups[0].files is None
        assert p.file_set.groups[0].unit == SaveUnitStatement(
            SAVE_UNIT_MEMBERS, ("01-GALE-a.gci", "01-GALE-b.GCI"), None
        )
        (membership,) = [c for c in p.caveats if c.code == atlas.CAVEAT_FILE_MEMBERSHIP_BY_HEADER]
        assert membership.data["game_code"] == "GALE"
        assert "save_id" not in membership.data

    def test_a_fresh_card_states_the_folder_with_no_members(self):
        p = _answer("gc", discs=GALE)
        assert p.needs == ()
        assert p.file_set.groups[0].unit == SaveUnitStatement(SAVE_UNIT_MEMBERS, (), None)

    def test_a_folder_that_cannot_be_listed_keeps_the_game_code_hole_and_states_no_unit(self):
        p = _answer("gc", discs=GALE, dirs=(USA_A,), unlistable=(USA_A,))
        assert p.dir == USA_A
        assert p.needs == (atlas.HOLE_SAVE_ID,)
        assert p.file_set.groups[0].unit is None

    def test_a_gci_whose_bytes_cannot_be_read_establishes_no_members(self):
        p = _answer("gc", discs=GALE, files={f"{USA_A}/01-GALE-a.gci": {"size": 8256}})
        assert p.needs == (atlas.HOLE_SAVE_ID,)
        assert p.file_set.groups[0].unit is None

    def test_a_raw_card_stays_one_shared_file_per_region(self):
        p = _answer("gc", discs=GALE, files={f"{RD}/config/dolphin-emu/Dolphin.ini": "[Core]\nSlotA = 1\n"})
        assert len(p.file_set.groups) == 3
        assert all(g.unit is None for g in p.file_set.groups)

    def test_a_disc_whose_region_nothing_establishes_keeps_both_holes(self):
        p = _answer("gc", discs={ROM_GC: _disc("gc", "GALX01")})
        assert p.needs == (atlas.HOLE_REGION, atlas.HOLE_SAVE_ID)
        assert all(g.unit is None for g in p.file_set.groups)

    @pytest.mark.parametrize("discs", [{}, {ROM_GC: "not-a-disc"}, {ROM_GC: _disc("wii", "RSBE01")}],
                             ids=["undeclared", "not-a-disc", "a-wii-disc"])  # fmt: skip
    def test_a_content_path_that_is_no_gamecube_disc_keeps_both_holes(self, discs):
        p = _answer("gc", discs=discs)
        assert p.needs == (atlas.HOLE_REGION, atlas.HOLE_SAVE_ID)
        assert all(g.unit is None for g in p.file_set.groups)

    def test_no_content_path_is_the_answer_without_a_disc(self):
        p = _answer("gc", discs=GALE, no_content=True)
        assert p.needs == (atlas.HOLE_REGION, atlas.HOLE_SAVE_ID)
        assert all(g.unit is None for g in p.file_set.groups)

    def test_a_configured_folder_is_narrowed_to_the_regions_directory(self):
        p = _answer(
            "gc",
            discs={ROM_GC: _disc("gc", "GALP01", container="plain", region=2)},
            files={f"{RD}/config/dolphin-emu/Dolphin.ini": "[Core]\nGCIFolderAPath = /mnt/cards/USA\n"},
        )
        assert p.dir == "/mnt/cards/EUR"
        assert p.needs == ()

    def test_the_statement_serializes_on_its_group(self):
        p = _answer("gc", discs=GALE, files={f"{USA_A}/01-GALE-a.gci": _gci(b"GALE")})
        group = savefile_placement_contract(p)["file_set"]["groups"][0]
        assert group["unit"] == {"kind": "own-files", "members": ["01-GALE-a.gci"], "archive_folder": None, "prefix": None,
                                 "archive_roots": None}


class TestTheStandaloneWiiAnswer:
    def test_the_tickets_title_is_its_own_folder_whole(self):
        p = _answer("wii", discs={ROM_WII: _disc("wii", "RSBE01", container="plain", title="0001000052534245")})
        directory = f"{STANDALONE_NAND}/title/00010000/52534245"
        assert p.dir == directory
        assert p.needs == ()
        (group,) = p.file_set.groups
        assert (group.dir, group.files, group.granularity) == (directory, None, atlas.GRANULARITY_PER_GAME_DIRECTORY)
        assert group.unit == SaveUnitStatement(SAVE_UNIT_FOLDER, None, "52534245")
        assert not [c for c in p.caveats if "save_id" in c.data]
        assert atlas.CAVEAT_SAVE_ID_CANDIDATES not in [c.code for c in p.caveats]

    def test_without_a_ticket_the_nands_one_title_of_the_game_code_names_the_high_word(self):
        p = _answer(
            "wii",
            discs={ROM_WII: _disc("wii", "RFNE01")},
            dirs=(f"{STANDALONE_NAND}/title/00010004/52464e45/data", f"{STANDALONE_NAND}/title/00010000/aaaaaaaa"),
        )
        assert p.dir == f"{STANDALONE_NAND}/title/00010004/52464e45"
        assert p.file_set.groups[0].unit == SaveUnitStatement(SAVE_UNIT_FOLDER, None, "52464e45")

    def test_without_a_ticket_a_nand_holding_only_the_channel_category_names_it(self):
        p = _answer("wii", discs={ROM_WII: _disc("wii", "RFNE01")},
                    dirs=(f"{STANDALONE_NAND}/title/00010004/52464e45/data",))  # fmt: skip
        assert p.dir == f"{STANDALONE_NAND}/title/00010004/52464e45"
        assert p.needs == ()
        assert not [source for source in p.sources if "disc-title default" in source]

    @pytest.mark.parametrize(
        "dirs",
        [(), (f"{STANDALONE_NAND}/title/00010000/52464e45", f"{STANDALONE_NAND}/title/00010004/52464e45")],
        ids=["never-booted", "two-categories"],
    )
    def test_without_a_ticket_a_low_word_the_nand_does_not_single_out_takes_the_disc_title_category(self, dirs):
        p = _answer("wii", discs={ROM_WII: _disc("wii", "RFNE01")}, dirs=dirs)
        directory = f"{STANDALONE_NAND}/title/00010000/52464e45"
        assert (p.dir, p.needs) == (directory, ())
        assert p.file_set.groups[0].dir == directory
        assert p.file_set.groups[0].unit == SaveUnitStatement(SAVE_UNIT_FOLDER, None, "52464e45")
        (derived,) = [source for source in p.sources if "disc-title default" in source]
        assert "Formats.h:33-41" in derived
        assert "ES.cpp:834-874, Boot_BS2Emu.cpp:621 at dolphin 2609" in derived

    def test_a_nand_that_cannot_be_listed_keeps_the_hole_rather_than_deriving_the_category(self):
        p = _answer("wii", discs={ROM_WII: _disc("wii", "RFNE01")}, unlistable=(f"{STANDALONE_NAND}/title",))
        assert atlas.HOLE_SAVE_ID in p.needs
        assert all(group.unit is None for group in p.file_set.groups)
        assert not [source for source in p.sources if "disc-title default" in source]

    def test_a_ticketed_disc_states_no_derived_category(self):
        p = _answer("wii", discs={ROM_WII: _disc("wii", "RSBE01", container="plain", title="0001000452534245")})
        assert p.dir == f"{STANDALONE_NAND}/title/00010004/52534245"
        assert not [source for source in p.sources if "disc-title default" in source]

    @pytest.mark.parametrize("discs", [{}, {ROM_WII: "not-a-disc"}, {ROM_WII: _disc("gc", "GALE01")}],
                             ids=["undeclared", "not-a-disc", "a-gamecube-disc"])  # fmt: skip
    def test_a_content_path_that_is_no_wii_disc_keeps_the_hole(self, discs):
        p = _answer("wii", discs=discs)
        assert p.dir == f"{STANDALONE_NAND}/title/<save_id>/data"
        assert p.file_set.groups[0].unit is None


class TestTheCoresAnswer:
    """The libretro core reads its tree through the same code, so its answers carry the statement too."""

    def test_the_cores_gamecube_answer_names_its_region_folder_and_members(self):
        folder = f"{SAVES}/gc/User/GC/USA/Card A"
        p = _answer("gc", core=True, discs=GALE, files={f"{folder}/01-GALE-a.gci": _gci(b"GALE")})
        assert p.dir == folder
        assert p.needs == ()
        assert p.file_set.groups[0].unit == SaveUnitStatement(SAVE_UNIT_MEMBERS, ("01-GALE-a.gci",), None)
        (membership,) = [c for c in p.caveats if c.code == atlas.CAVEAT_FILE_MEMBERSHIP_BY_HEADER]
        assert membership.data["core"] == "dolphin"

    def test_the_cores_wii_answer_names_the_tickets_title(self):
        p = _answer("wii", core=True, discs={ROM_WII: _disc("wii", "RSBE01", container="wbfs",
                                                                 title="0001000052534245")})  # fmt: skip
        assert p.dir == f"{SAVES}/wii/User/Wii/title/00010000/52534245"
        assert p.needs == ()
        assert p.file_set.groups[0].unit == SaveUnitStatement(SAVE_UNIT_FOLDER, None, "52534245")


class TestOnlyTheReadingsThatEstablishItStateIt:
    """A per-game-directory granularity is not the statement: RPCS3's names every title's parent."""

    def _answers(self) -> list[tuple[str | None, dict[str, Any]]]:
        answers: list[tuple[str | None, dict[str, Any]]] = []
        for path in sorted(Path(__file__).parent.parent.joinpath("vectors", "machines").glob("*.json")):
            for vector in json.loads(path.read_text(encoding="utf-8"))["vectors"]:
                query = vector["input"].get("entry_savefile_query") or {}
                answer = vector["expected"].get("entry_savefile_location") or {}
                answers.append((query.get("system"), answer))
        return answers

    def test_no_vector_outside_the_reading_systems_states_a_unit(self):
        # GameCube and Wii by Dolphin's disc reads; Switch by Eden's, which
        # names the application and the active user; PSP by PPSSPP's, which
        # names the game's id; 3DS by Azahar's, which names the title.
        stated = [
            system
            for system, answer in self._answers()
            for group in (answer.get("file_set") or {}).get("groups", [])
            if group["unit"] is not None
        ]
        assert stated and set(stated) <= {"gc", "wii", "switch", "psp", "n3ds"}

    @pytest.mark.parametrize("system", ["ps3", "wiiu"])
    def test_rpcs3_and_cemu_answer_per_game_directories_with_no_unit(self, system):
        groups = [
            group
            for asked, answer in self._answers()
            if asked == system
            for group in (answer.get("file_set") or {}).get("groups", [])
            if group["granularity"] == atlas.GRANULARITY_PER_GAME_DIRECTORY
        ]
        assert groups
        assert all(group["unit"] is None for group in groups)
