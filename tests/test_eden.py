"""A standalone Eden AppImage: detection, its one entry, its launch, and the save of one Switch game.

Three layers, each proven on its own: the pure readings (:mod:`atlas.switch_title`
names an application id, :mod:`atlas.eden` reads ``qt-config.ini`` and
``profiles.dat``), the real machine's two byte reads over synthetic files, and
the handle over fixture machines. The conformance vectors
(``vectors/machines/eden-appimage.json``) hold the serialized answers.
"""

from __future__ import annotations

import io
import struct
from collections.abc import Mapping
from pathlib import Path

import pytest

import atlas
from atlas import eden, switch_title
from atlas.machine import (
    EDEN_PROFILES_MISSING,
    EDEN_PROFILES_OK,
    EDEN_PROFILES_SHORT,
    SWITCH_PACKAGE_MISSING,
    SWITCH_PACKAGE_NOT_A_PACKAGE,
    SWITCH_PACKAGE_OK,
    FixtureMachine,
    RealMachine,
)
from atlas.placement import SAVE_UNIT_FOLDER, SaveUnitStatement

HOME = "/home/deck"
APP = f"{HOME}/Applications/Eden-Linux-v0.2.1-amd64.AppImage"
NIGHTLY = f"{HOME}/AppImages/eden-nightly.AppImage"
NAND = f"{HOME}/.local/share/eden/nand"
PROFILES = f"{NAND}/system/save/8000000000000010/su/avators/profiles.dat"
CONFIG = f"{HOME}/.config/eden/qt-config.ini"
SAVES = f"{NAND}/user/save/0000000000000000"
USER_A = "00112233445566778899aabbccddeeff"
FOLDER_A = "FFEEDDCCBBAA99887766554433221100"
USER_B = "0f1e2d3c4b5a69788796a5b4c3d2e1f0"
FOLDER_B = "F0E1D2C3B4A5968778695A4B3C2D1E0F"
ZELDA = "/roms/switch/Zelda [01007EF00011E000].nsp"
ZELDA_ID = "01007EF00011E000"
BLOB = {"size": 4096}


def _pfs0(names: list[str]) -> bytes:
    table = b"".join(name.encode() + b"\0" for name in names)
    offsets, at = [], 0
    for name in names:
        offsets.append(at)
        at += len(name.encode()) + 1
    entries = b"".join(struct.pack("<QQII", 0, 0, offset, 0) for offset in offsets)
    return b"PFS0" + struct.pack("<II4x", len(names), len(table)) + entries + table


def _profiles(*uuids: str, gaps: tuple[int, ...] = ()) -> bytes:
    records = [bytes.fromhex(uuid) for uuid in uuids]
    for index in gaps:
        records.insert(index, bytes(16))
    data = bytearray(eden.PROFILES_SIZE)
    for slot, uuid in enumerate(records):
        start = 0x10 + slot * 0xC8
        data[start : start + 16] = uuid
    return bytes(data)


def _machine(files: Mapping[str, object] | None = None, **kwargs: object) -> FixtureMachine:
    return FixtureMachine({APP: BLOB, PROFILES: BLOB, **(files or {})}, **kwargs)  # pyright: ignore[reportArgumentType]


def _handle(machine: FixtureMachine, **kwargs: object) -> atlas.EdenAppImage:
    installs = atlas.detect(HOME, machine, **kwargs)  # pyright: ignore[reportArgumentType]
    [handle] = [i for i in installs if isinstance(i, atlas.EdenAppImage)]
    return handle


def _save(handle: atlas.EdenAppImage, content_path: str | None = None) -> atlas.SavefilePlacement:
    [entry] = handle.emulators_for("switch").entries
    answer = entry.savefile_location(content_path=content_path)
    assert isinstance(answer, atlas.SavefilePlacement), answer
    return answer


class TestNamingTheApplication:
    @pytest.mark.parametrize(
        ("title_id", "application"),
        [
            (0x01007EF00011E000, 0x01007EF00011E000),  # the application itself
            (0x01007EF00011E800, 0x01007EF00011E000),  # its patch: app + 0x800
            (0x01007EF00011EC00, 0x01007EF00011E000),  # a delta fragment: app + 0xC00
            (0x01007EF00011F001, 0x01007EF00011E000),  # add-on 1: (app & ~0xFFF) + 0x1000 + 1
            (0x01007EF00011F7D0, 0x01007EF00011E000),  # add-on 2000, the last index
        ],
    )
    def test_every_title_kind_goes_back_to_its_application(self, title_id: int, application: int):
        assert switch_title.application_id(title_id) == application

    @pytest.mark.parametrize("title_id", [0x01007EF00011E7D1, 0x01007EF00011E900, 0x0000000000000001])
    def test_an_id_no_numbering_derives_names_no_application(self, title_id: int):
        assert switch_title.application_id(title_id) is None

    def test_tags_are_read_out_of_a_name_in_either_case(self):
        assert switch_title.tagged_title_ids("Game [01007ef00011e000][v0] [0100ABCD12340800].nsp") == (
            0x01007EF00011E000,
            0x0100ABCD12340800,
        )

    @pytest.mark.parametrize("name", ["Game (01007EF00011E000).nsp", "Game [01007EF00011E00].nsp", "Game [v0].nsp"])
    def test_anything_but_a_bracketed_sixteen_digit_user_title_is_no_tag(self, name: str):
        assert switch_title.tagged_title_ids(name) == ()

    def test_the_spelling_is_sixteen_uppercase_digits(self):
        assert switch_title.spelled(0x01007EF00011E000) == ZELDA_ID

    def test_tickets_name_their_title_ids(self):
        names = ("abc.nca", "01007EF00011E8000000000000000004.tik", "x.cnmt.xml", "not-a-ticket.tik")
        assert switch_title.ticket_title_ids(names) == (0x01007EF00011E800,)


class TestReadingAPfs0Header:
    def test_the_names_come_back_in_entry_order(self):
        names = ["0123.nca", "01007EF00011E0000000000000000004.tik", "0123.cnmt.xml"]
        assert switch_title.read_pfs0_names(io.BytesIO(_pfs0(names) + b"\xff" * 64)) == tuple(names)

    @pytest.mark.parametrize(
        "data",
        [
            b"HFS0" + b"\0" * 12,  # a cartridge's partition, not a package
            b"PFS0\x01\0\0\0",  # cut inside the header
            _pfs0(["a.nca", "b.tik"])[:-3],  # cut inside the string table
            b"PFS0" + struct.pack("<II4x", 0xFFFFFF, 0),  # a header past the bound
            b"PFS0" + struct.pack("<II4x", 1, 4) + struct.pack("<QQII", 0, 0, 9, 0) + b"abc\0",  # name off the table
            b"",
        ],
    )
    def test_garbage_is_no_package(self, data: bytes):
        with pytest.raises(switch_title.SwitchPackageUnread):
            switch_title.read_pfs0_names(io.BytesIO(data))


class TestReadingEdensFiles:
    def test_profiles_list_the_users_in_the_order_eden_counts_them(self):
        data = _profiles(USER_A, USER_B, gaps=(0,))
        assert eden.parse_profiles(data) == (bytes.fromhex(USER_A), bytes.fromhex(USER_B))

    def test_a_short_file_is_no_list(self):
        with pytest.raises(eden.ProfilesUnread):
            eden.parse_profiles(_profiles(USER_A)[:-1])

    def test_a_user_folder_is_the_high_word_then_the_low_word(self):
        assert eden.user_folder(bytes.fromhex(USER_A)) == FOLDER_A
        assert eden.user_folder(bytes.fromhex(USER_B)) == FOLDER_B
        assert eden.uuid_raw(bytes.fromhex(USER_A)) == USER_A

    @pytest.mark.parametrize(
        ("text", "index"),
        [
            ("", 0),
            ("[System]\ncurrent_user=3\n", 0),  # no \default=false: the default stands
            ("[System]\ncurrent_user\\default=false\ncurrent_user=3\n", 3),
            ("[system]\nCURRENT_USER\\DEFAULT=No\ncurrent_user=\"2\"\n", 2),  # case-insensitive, quotes dropped
            ("[System]\ncurrent_user\\default=false\ncurrent_user= 5abc\n", 5),  # stoll reads the leading digits
            ("[System]\ncurrent_user\\default=false\ncurrent_user=abc\n", 0),
            ("[System]\ncurrent_user\\default=off\ncurrent_user=-1\n", -1),
            ("[System]\ncurrent_user\\default=false\ncurrent_user=1\ncurrent_user=4\n", 4),  # the last one wins
        ],
    )
    def test_the_current_user_index_is_read_as_eden_reads_it(self, text: str, index: int):
        assert eden.current_user_index(eden.ini_values(text)) == index

    @pytest.mark.parametrize(
        ("index", "users", "slot"),
        [(0, 1, 0), (2, 3, 2), (5, 2, 0), (-1, 2, 0), (9, 8, 7), (0, 0, None)],
    )
    def test_the_slot_is_clamped_and_falls_back_to_the_first(self, index: int, users: int, slot: int | None):
        assert eden.active_user_slot(index, users) == slot

    def test_the_data_storage_section_is_its_escaped_name(self):
        values = eden.ini_values('[Data%20Storage]\nnand_directory="/mnt/nand"\n[Data Storage]\nsave_directory=/x\n')
        assert eden.ini_string(values, eden.SECTION_DATA_STORAGE, eden.KEY_NAND_DIRECTORY) == "/mnt/nand"
        assert eden.ini_string(values, eden.SECTION_DATA_STORAGE, eden.KEY_SAVE_DIRECTORY) is None


class TestTheRealMachineReads:
    def test_a_package_header(self, tmp_path: Path):
        package = tmp_path / "game.nsp"
        package.write_bytes(_pfs0(["01007EF00011E0000000000000000004.tik"]) + b"\0" * 128)
        read = RealMachine().read_switch_package(str(package))
        assert (read.status, read.names) == (SWITCH_PACKAGE_OK, ("01007EF00011E0000000000000000004.tik",))

    def test_a_file_that_is_no_package(self, tmp_path: Path):
        xci = tmp_path / "game.xci"
        xci.write_bytes(b"HEAD" * 64)
        assert RealMachine().read_switch_package(str(xci)).status == SWITCH_PACKAGE_NOT_A_PACKAGE
        assert RealMachine().read_switch_package(str(tmp_path / "none.nsp")).status == SWITCH_PACKAGE_MISSING

    def test_a_profile_list(self, tmp_path: Path):
        profiles = tmp_path / "profiles.dat"
        profiles.write_bytes(_profiles(USER_A))
        read = RealMachine().read_eden_profiles(str(profiles))
        assert (read.status, read.users) == (EDEN_PROFILES_OK, (bytes.fromhex(USER_A),))

    def test_a_short_or_missing_profile_list(self, tmp_path: Path):
        profiles = tmp_path / "profiles.dat"
        profiles.write_bytes(b"\0" * 0x10)
        assert RealMachine().read_eden_profiles(str(profiles)).status == EDEN_PROFILES_SHORT
        assert RealMachine().read_eden_profiles(str(tmp_path / "none.dat")).status == EDEN_PROFILES_MISSING


class TestDetection:
    def test_an_appimage_in_a_searched_folder_is_found(self):
        handle = _handle(_machine(eden_profiles={PROFILES: [USER_A]}))
        assert (handle.kind, handle.appimage(), handle.health().ok) == ("eden_appimage", APP, True)

    @pytest.mark.parametrize(
        "path",
        [
            f"{HOME}/AppImages/EDEN.APPIMAGE",
            f"{HOME}/Emulation/tools/eden.AppImage",
            f"{HOME}/.local/bin/Eden-nightly.AppImage",
        ],
    )
    def test_every_searched_folder_is_searched_case_insensitively(self, path: str):
        assert _handle(FixtureMachine({path: BLOB})).appimage() == path

    def test_nothing_found_is_no_installation(self):
        files = {f"{HOME}/Applications/Ryujinx.AppImage": BLOB, f"{HOME}/Downloads/Eden.AppImage": BLOB}
        assert atlas.detect(HOME, FixtureMachine(files)) == []  # pyright: ignore[reportArgumentType]

    def test_a_directory_named_like_one_is_not_one(self):
        assert atlas.detect(HOME, FixtureMachine({}, dirs=[f"{HOME}/Applications/Eden.AppImage"])) == []

    def test_a_named_appimage_wins_over_the_search(self):
        named = "/opt/eden/Eden.AppImage"
        handle = _handle(_machine({named: BLOB, NIGHTLY: BLOB}), eden_appimage=named)
        assert (handle.appimage(), handle.health().ok) == (named, True)

    def test_a_named_appimage_that_is_not_there_is_a_handle_that_says_so(self):
        handle = _handle(FixtureMachine({}), eden_appimage="/opt/eden/Eden.AppImage")
        assert handle.appimage() is None
        assert handle.health().codes == (atlas.HEALTH_ISSUE_MARKER_MISSING,)
        [entry] = handle.emulators_for("switch").entries
        assert entry.availability == "not-installed"

    def test_a_relative_name_is_refused(self):
        with pytest.raises(ValueError, match="absolute"):
            atlas.detect(HOME, FixtureMachine({}), eden_appimage="Eden.AppImage")  # pyright: ignore[reportArgumentType]

    def test_two_found_are_never_picked_between(self):
        handle = _handle(_machine({NIGHTLY: BLOB}))
        assert handle.appimage() is None
        [issue] = handle.health().issues
        assert (issue.code, issue.data["candidates"]) == (atlas.HEALTH_ISSUE_APPIMAGE_AMBIGUOUS, (NIGHTLY, APP))
        [entry] = handle.emulators_for("switch").entries
        assert entry.availability == "unestablished"
        assert handle.launch_command("switch", ZELDA).command is None

    def test_the_aggregate_takes_the_name_too(self):
        named = "/opt/eden/Eden.AppImage"
        every = atlas.every_installation(HOME, FixtureMachine({named: BLOB}), eden_appimage=named)  # pyright: ignore[reportArgumentType]
        assert [i.kind for i in every.installations] == ["eden_appimage"]

    def test_the_handle_answers_every_public_question_as_one(self):
        assert isinstance(_handle(_machine()), atlas.Installation)


class TestTheEntryAndItsLaunch:
    def test_switch_has_one_startable_standalone_entry(self):
        handle = _handle(_machine())
        [entry] = handle.emulators_for("switch").entries
        assert (entry.kind, entry.emulator, entry.core_so, entry.availability) == ("standalone", "EDEN", None, "startable")
        assert entry.launcher is not None and entry.launcher.path == APP
        assert handle.emulators_for("gb").entries == ()
        assert handle.systems().systems == ("switch",)

    def test_the_command_is_the_appimage_with_f_g_in_its_own_folder(self):
        command = _handle(_machine()).launch_command("switch", ZELDA).command
        assert command is not None
        assert (command.program.variant, command.program.path) == ("appimage", APP)
        assert command.arguments == ("-f", "-g", ZELDA)
        assert command.environment == () and command.distribution_environment == ()
        assert command.working_folder is not None
        assert command.working_folder.host_path == f"{HOME}/Applications"

    def test_a_label_the_system_does_not_carry_is_refused(self):
        answer = _handle(_machine()).launch_command("switch", ZELDA, label="Ryujinx")
        assert answer.command is None
        assert atlas.CAVEAT_LAUNCH_COMMAND_LABEL_UNKNOWN in [c.code for c in answer.caveats]

    @pytest.mark.parametrize(("path", "verdict"), [(ZELDA, "launchable"), ("/roms/switch/Game.zip", "unknown")])
    def test_launchability_is_eden_loaders_names_and_never_its_content(self, path: str, verdict: str):
        assert _handle(_machine()).launchable("switch", path).verdict == verdict

    def test_firmware_is_stated_unsupported(self):
        [core] = _handle(_machine()).firmware_for_system("switch").cores
        assert (core.declaration, core.emulator, core.requirements_met) == ("unsupported", "EDEN", None)


class TestTheSave:
    def test_the_tag_and_the_only_user_name_the_games_own_folder(self):
        p = _save(_handle(_machine({ZELDA: BLOB}, eden_profiles={PROFILES: [USER_A]})), ZELDA)
        assert p.dir == f"{SAVES}/{FOLDER_A}/{ZELDA_ID}"
        assert p.needs == ()
        assert p.file_set.groups[0].unit == SaveUnitStatement(SAVE_UNIT_FOLDER, None, ZELDA_ID)

    def test_current_user_picks_among_the_listed_users(self):
        config = "[System]\ncurrent_user\\default=false\ncurrent_user=1\n"
        machine = _machine({ZELDA: BLOB, CONFIG: config}, eden_profiles={PROFILES: [USER_A, USER_B]})
        assert _save(_handle(machine), ZELDA).dir == f"{SAVES}/{FOLDER_B}/{ZELDA_ID}"

    def test_an_index_no_user_holds_falls_back_to_the_first(self):
        config = "[System]\ncurrent_user\\default=false\ncurrent_user=6\n"
        machine = _machine({ZELDA: BLOB, CONFIG: config}, eden_profiles={PROFILES: [USER_A, USER_B]})
        assert _save(_handle(machine), ZELDA).dir == f"{SAVES}/{FOLDER_A}/{ZELDA_ID}"

    @pytest.mark.parametrize("profiles", ["short", []])
    def test_a_list_eden_would_replace_with_a_fresh_user_keeps_the_hole(self, profiles: object):
        machine = _machine({ZELDA: BLOB}, eden_profiles={PROFILES: profiles}, dirs=[f"{SAVES}/{FOLDER_A}"])
        p = _save(_handle(machine), ZELDA)
        assert p.needs == (atlas.HOLE_PROFILE_ID,)
        assert p.dir == f"{SAVES}/<profile_id>/{ZELDA_ID}"
        assert p.file_set.groups[0].unit is None
        [refusal] = [c for c in p.caveats if c.code == atlas.CAVEAT_ACTIVE_PROFILE_UNESTABLISHED]
        assert refusal.data["users"] == (FOLDER_A,)

    def test_a_missing_list_keeps_the_hole_whatever_folders_are_there(self):
        machine = FixtureMachine({APP: BLOB, ZELDA: BLOB}, dirs=[f"{SAVES}/{FOLDER_A}"])
        assert _save(_handle(machine), ZELDA).needs == (atlas.HOLE_PROFILE_ID,)

    def test_an_unreadable_list_beside_one_user_folder_names_that_user(self):
        machine = FixtureMachine(
            {APP: BLOB, PROFILES: {"status": "unreadable"}, ZELDA: BLOB},
            dirs=[f"{SAVES}/{FOLDER_A}", f"{SAVES}/{eden.SYSTEM_USER_FOLDER}"],
        )
        p = _save(_handle(machine), ZELDA)
        assert (p.dir, p.needs) == (f"{SAVES}/{FOLDER_A}/{ZELDA_ID}", ())
        assert any("derived rather than read" in source for source in p.sources)

    def test_an_unreadable_list_beside_two_user_folders_keeps_the_hole(self):
        machine = FixtureMachine(
            {APP: BLOB, PROFILES: {"status": "unreadable"}, ZELDA: BLOB},
            dirs=[f"{SAVES}/{FOLDER_A}", f"{SAVES}/{FOLDER_B}"],
        )
        assert _save(_handle(machine), ZELDA).needs == (atlas.HOLE_PROFILE_ID,)

    def test_no_content_keeps_the_title_hole_and_says_how_to_fill_it(self):
        p = _save(_handle(_machine(eden_profiles={PROFILES: [USER_A]})))
        assert (p.dir, p.needs) == (f"{SAVES}/{FOLDER_A}/<save_id>", (atlas.HOLE_SAVE_ID,))
        [names] = [c for c in p.caveats if c.code == atlas.CAVEAT_FILE_NAMES_UNESTABLISHED]
        assert names.data["save_id_spelling"] == "path-segment"

    def test_an_add_on_tag_names_its_application(self):
        dlc = "/roms/switch/Zelda DLC [01007EF00011F001].nsp"
        p = _save(_handle(_machine({dlc: BLOB}, eden_profiles={PROFILES: [USER_A]})), dlc)
        assert p.dir.endswith(f"/{ZELDA_ID}")

    def test_the_package_header_names_an_untagged_package(self):
        game = "/roms/switch/untagged.nsp"
        machine = _machine(
            {game: BLOB},
            eden_profiles={PROFILES: [USER_A]},
            switch_packages={game: ["a.nca", "01007EF00011E8000000000000000004.tik"]},
        )
        assert _save(_handle(machine), game).dir == f"{SAVES}/{FOLDER_A}/{ZELDA_ID}"

    def test_a_package_ticketing_two_applications_names_none(self):
        game = "/roms/switch/bundle.nsp"
        tickets = ["01007EF00011E0000000000000000004.tik", "0100152000022000000000000000000a.tik"]
        machine = _machine({game: BLOB}, eden_profiles={PROFILES: [USER_A]}, switch_packages={game: tickets})
        assert _save(_handle(machine), game).needs == (atlas.HOLE_SAVE_ID,)

    def test_a_folders_other_files_name_the_game_where_the_content_does_not(self):
        base = "/roms/switch/Mario Kart/base.xci"
        update = "/roms/switch/Mario Kart/update [0100152000022800].nsp"
        machine = _machine({base: BLOB, update: BLOB}, eden_profiles={PROFILES: [USER_A]})
        assert _save(_handle(machine), base).dir == f"{SAVES}/{FOLDER_A}/0100152000022000"

    def test_neighbours_of_two_games_name_none(self):
        game = "/roms/switch/untitled.xci"
        files = {game: BLOB, "/roms/switch/A [01007EF00011E000].nsp": BLOB, "/roms/switch/B [0100152000022000].nsp": BLOB}
        assert _save(_handle(_machine(files, eden_profiles={PROFILES: [USER_A]})), game).needs == (atlas.HOLE_SAVE_ID,)

    def test_a_configured_nand_moves_the_saves_and_the_user_list(self):
        nand = "/mnt/sd/nand"
        config = f'[Data%20Storage]\nnand_directory="{nand}"\n'
        moved = f"{nand}/system/save/8000000000000010/su/avators/profiles.dat"
        machine = FixtureMachine({APP: BLOB, CONFIG: config, moved: BLOB, ZELDA: BLOB}, eden_profiles={moved: [USER_A]})
        p = _save(_handle(machine), ZELDA)
        assert p.dir == f"{nand}/user/save/0000000000000000/{FOLDER_A}/{ZELDA_ID}"

    def test_a_configured_nand_that_is_no_directory_is_ignored(self):
        config = '[Data%20Storage]\nnand_directory="/mnt/gone"\n'
        machine = _machine({CONFIG: config, ZELDA: BLOB}, eden_profiles={PROFILES: [USER_A]})
        assert _save(_handle(machine), ZELDA).dir == f"{SAVES}/{FOLDER_A}/{ZELDA_ID}"

    def test_the_layout_eden_prefers_wins_where_it_exists(self):
        future = f"{NAND}/user/save/account/{USER_A}/{ZELDA_ID}/0"
        machine = _machine({ZELDA: BLOB}, eden_profiles={PROFILES: [USER_A]}, dirs=[future])
        p = _save(_handle(machine), ZELDA)
        assert p.dir == future
        assert p.file_set.groups[0].unit == SaveUnitStatement(SAVE_UNIT_FOLDER, None, ZELDA_ID)

    def test_a_portable_tree_beside_the_appimage_is_the_data_root(self):
        portable = f"{HOME}/Applications/user"
        moved = f"{portable}/nand/system/save/8000000000000010/su/avators/profiles.dat"
        machine = FixtureMachine({APP: BLOB, moved: BLOB, ZELDA: BLOB}, eden_profiles={moved: [USER_A]})
        handle = _handle(machine)
        assert handle.root() == portable
        assert _save(handle, ZELDA).dir == f"{portable}/nand/user/save/0000000000000000/{FOLDER_A}/{ZELDA_ID}"

    def test_two_appimages_reading_different_trees_refuse_the_save(self):
        machine = _machine({NIGHTLY: BLOB, ZELDA: BLOB}, dirs=[f"{HOME}/AppImages/user"])
        [entry] = _handle(machine).emulators_for("switch").entries
        answer = entry.savefile_location(content_path=ZELDA)
        assert isinstance(answer, atlas.Unresolved) and answer.code == atlas.UNRESOLVED_APPIMAGE_AMBIGUOUS

    def test_the_core_route_answers_the_same_save_or_refuses_a_core(self):
        handle = _handle(_machine({ZELDA: BLOB}, eden_profiles={PROFILES: [USER_A]}))
        direct = handle.savefile_location(content_path=ZELDA, system="switch")
        assert isinstance(direct, atlas.SavefilePlacement) and direct.dir == f"{SAVES}/{FOLDER_A}/{ZELDA_ID}"
        refused = handle.savefile_location(core_so="mgba_libretro.so")
        assert isinstance(refused, atlas.Unresolved) and refused.code == "core-not-installed"


class TestReadingWhereExtrasLie:
    """``[Paths] gamedirs``, ``external_content_dirs`` and ``ext_content_from_game_dirs``, read the way Eden reads them."""

    def test_game_dir_rows_keep_their_order_and_deep_scan(self):
        values = eden.ini_values(
            "[Paths]\n"
            "gamedirs\\size=2\n"
            "gamedirs\\1\\path=SDMC\n"
            "gamedirs\\2\\path=\"/roms/my switch\"\n"
            "gamedirs\\2\\deep_scan\\default=false\n"
            "gamedirs\\2\\deep_scan=true\n"
        )
        assert eden.game_dirs(values) == (eden.GameDir("SDMC", False), eden.GameDir("/roms/my switch", True))

    def test_a_deep_scan_whose_default_companion_is_true_is_off_whatever_it_says(self):
        values = eden.ini_values(
            "[Paths]\ngamedirs\\size=1\ngamedirs\\1\\path=/roms\n"
            "gamedirs\\1\\deep_scan\\default=true\ngamedirs\\1\\deep_scan=true\n"
        )
        assert eden.game_dirs(values) == (eden.GameDir("/roms", False),)

    def test_no_row_puts_the_installed_content_rows_in_and_carries_the_old_root_over(self):
        assert [row.path for row in eden.game_dirs({})] == ["SDMC", "UserNAND", "SysNAND"]
        values = eden.ini_values(
            "[Paths]\ngameListRootDir\\default=false\ngameListRootDir=/old//roms\n"
            "gameListDeepScan\\default=false\ngameListDeepScan=true\n"
        )
        assert eden.game_dirs(values)[-1] == eden.GameDir("/old/roms", True)

    def test_external_content_dirs_drop_quotes_and_skip_empty_rows(self):
        values = eden.ini_values(
            "[Paths]\nexternal_content_dirs\\size=3\nexternal_content_dirs\\1\\path=\"/a b\"\n"
            "external_content_dirs\\2\\path=\nexternal_content_dirs\\3\\path=/c\n"
        )
        assert eden.external_content_dirs(values) == ("/a b", "/c")

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("", True),
            ("[Data%20Storage]\next_content_from_game_dirs=false\n", True),
            ("[Data%20Storage]\next_content_from_game_dirs\\default=false\next_content_from_game_dirs=false\n", False),
            ("[Data%20Storage]\next_content_from_game_dirs\\default=false\next_content_from_game_dirs=True\n", False),
            ("[Data%20Storage]\next_content_from_game_dirs\\default=false\next_content_from_game_dirs=\n", True),
            ("[Data%20Storage]\next_content_from_game_dirs\\default=false\next_content_from_game_dirs=true\n", True),
        ],
    )
    def test_ext_content_from_game_dirs_is_a_generic_setting_true_only_as_true(self, text, expected):
        assert eden.ext_content_from_game_dirs(eden.ini_values(text)) is expected

    @pytest.mark.parametrize(
        ("value", "expected"), [("3", 3), ("0x10", 16), (" 2", 2), ("-1", -1), ("2x", 7), ("0x", 7), ("", 7), (None, 7)]
    )
    def test_an_array_size_reads_as_simpleini_reads_a_long(self, value, expected):
        assert eden.simpleini_long(value, 7) == expected


class TestTheExtrasEdenLoadsFromFolders:
    """The extra-content answer of the Eden entry, and of a system it has none for."""

    def _answer(self, config: str, content_path: str | None = None, files: Mapping[str, object] | None = None,
                **kwargs: object) -> atlas.ExtraContentAnswer:
        machine = _machine({CONFIG: config, **(files or {})})
        return _handle(machine, **kwargs).extra_content_folders("switch", content_path=content_path)

    def test_a_deep_scanned_game_folder_covers_a_game_in_a_folder_of_its_own(self):
        config = (
            "[Paths]\ngamedirs\\size=1\ngamedirs\\1\\path=/roms/switch\n"
            "gamedirs\\1\\deep_scan\\default=false\ngamedirs\\1\\deep_scan=true\n"
        )
        game = "/roms/switch/Zelda/Zelda.nsp"
        answer = self._answer(config, game, {game: BLOB})
        assert answer.loads_from_folders is True
        assert answer.folders == (atlas.ExtraContentFolder("/roms/switch", True, None, (".nsp", ".xci")),)
        assert answer.content_folder_scanned is True
        assert atlas.CAVEAT_CONTENT_OUTSIDE_EXTRA_CONTENT_FOLDERS not in {c.code for c in answer.caveats}

    def test_a_relative_folder_is_below_the_appimages_own_folder(self):
        config = "[Paths]\nexternal_content_dirs\\size=1\nexternal_content_dirs\\1\\path=dlc\n"
        answer = self._answer(config, f"{HOME}/Applications/dlc/Game/Game.nsp")
        assert answer.folders[0].path == f"{HOME}/Applications/dlc"
        assert answer.content_folder_scanned is True

    def test_a_sibling_that_shares_a_prefix_is_not_inside(self):
        config = "[Paths]\nexternal_content_dirs\\size=1\nexternal_content_dirs\\1\\path=/extras\n"
        answer = self._answer(config, "/extras-old/Game.nsp")
        assert answer.content_folder_scanned is False
        [outside] = [c for c in answer.caveats if c.code == atlas.CAVEAT_CONTENT_OUTSIDE_EXTRA_CONTENT_FOLDERS]
        assert outside.data == {"folder": "/extras-old"}

    def test_no_content_leaves_the_content_verdict_open_and_says_nothing_of_outside(self):
        config = "[Paths]\nexternal_content_dirs\\size=1\nexternal_content_dirs\\1\\path=/extras\n"
        answer = self._answer(config)
        assert answer.content_folder_scanned is None
        assert {c.code for c in answer.caveats} == {"arrangement-unverified"}

    def test_a_relative_folder_with_appimages_in_two_folders_is_not_placed(self):
        config = "[Paths]\nexternal_content_dirs\\size=1\nexternal_content_dirs\\1\\path=dlc\n"
        machine = _machine({CONFIG: config, NIGHTLY: BLOB})
        answer = _handle(machine).extra_content_folders("switch", content_path="/roms/switch/Game.nsp")
        assert answer.folders == ()
        assert answer.content_folder_scanned is None
        assert atlas.CAVEAT_APPIMAGE_AMBIGUOUS in {c.code for c in answer.caveats}

    def test_appimages_reading_different_trees_name_no_folder(self):
        machine = _machine({NIGHTLY: BLOB}, dirs=[f"{HOME}/AppImages/user"])
        answer = _handle(machine).extra_content_folders("switch", content_path=ZELDA)
        assert (answer.loads_from_folders, answer.folders, answer.content_folder_scanned) == (True, (), None)
        assert atlas.CAVEAT_APPIMAGE_AMBIGUOUS in {c.code for c in answer.caveats}

    def test_another_system_or_an_unknown_label_states_nothing(self):
        handle = _handle(_machine())
        other = handle.extra_content_folders("gba")
        assert (other.entry, other.loads_from_folders) == (None, None)
        assert atlas.CAVEAT_EXTRA_CONTENT_FOLDERS_UNSTATED in {c.code for c in other.caveats}
        unknown = handle.extra_content_folders("switch", label="Ryujinx")
        assert unknown.loads_from_folders is None
        assert "launch-command-label-unknown" in {c.code for c in unknown.caveats}

    def test_a_relative_content_path_is_refused_at_the_call(self):
        with pytest.raises(ValueError):
            _handle(_machine()).extra_content_folders("switch", content_path="roms/Game.nsp")
        with pytest.raises(ValueError):
            atlas.every_installation(HOME, _machine()).extra_content_folders("switch", content_path="roms/Game.nsp")

    def test_every_installation_asks_each_handle(self):
        [answer] = atlas.every_installation(HOME, _machine()).extra_content_folders("switch")
        assert answer.answer.loads_from_folders is True
