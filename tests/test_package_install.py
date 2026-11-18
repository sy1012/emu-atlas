"""RPCS3's package install: the metadata a package's header carries, and what an install takes and where it goes.

The answer end to end is held by ``vectors/machines/package-install.json``;
this module holds what a vector cannot isolate — the metadata walk over real
bytes, the reading over a view whose probes it controls, and the answer's own
consistency rules.
"""

from __future__ import annotations

import io
import os
import struct
from pathlib import Path

import pytest

import atlas
from atlas import package_install, ps3_title
from atlas.launch import PROBE_HIT, PROBE_MISS, PROBE_UNKNOWN, Probe
from atlas.machine import PS3_TITLE_OK, FixtureMachine, Ps3TitleResult, RealMachine

HDD0 = "/emu/dev_hdd0/"
GAMES = "/roms/ps3/Game"


def pkg(packets: list[tuple[int, bytes]], *, count: int | None = None, offset: int = 0xC0) -> bytes:
    """A PS3 package: its header, the content id at 0x30, and *packets* at *offset*."""
    head = bytearray(offset)
    head[0:4] = b"\x7fPKG"
    struct.pack_into(">HHII", head, 4, 0x8000, 0x0001, offset, len(packets) if count is None else count)
    content_id = b"UP0001-NPUB30000_00-GAME000000000001"
    head[0x30 : 0x30 + len(content_id)] = content_id
    body = b"".join(struct.pack(">II", packet_id, len(data)) + data for packet_id, data in packets)
    return bytes(head) + body


def read(data: bytes) -> ps3_title.Ps3TitleFacts:
    return ps3_title.read_facts(io.BytesIO(data), len(data))


class TestTheMetadata:
    def test_the_content_type_packet_is_read(self):
        facts = read(pkg([(0x1, b"\0\0\0\x03"), (0x2, b"\0\0\0\x05")]))
        assert (facts.title_id, facts.content_type, facts.install_dir) == ("NPUB30000", 5, None)

    def test_no_content_type_packet_is_the_default_zero(self):
        assert read(pkg([])).content_type == 0

    def test_an_install_directory_packet_names_the_folder_after_its_first_eight_bytes(self):
        facts = read(pkg([(0x2, b"\0\0\0\x04"), (0xA, b"\0" * 8 + b"NPUB30001\0\0\0")]))
        assert (facts.content_type, facts.install_dir) == (4, "NPUB30001")

    def test_a_later_packet_replaces_an_earlier_one(self):
        facts = read(pkg([(0x2, b"\0\0\0\x05"), (0x2, b"\0\0\0\x0b")]))
        assert facts.content_type == 0x0B

    def test_a_packet_of_a_size_rpcs3_refuses_is_passed_over(self):
        facts = read(pkg([(0x2, b"\0\0\0\0\x05"), (0xA, b"\0" * 8)]))
        assert (facts.content_type, facts.install_dir) == (0, None)

    def test_metadata_reaching_beyond_the_file_is_not_read(self):
        facts = read(pkg([(0x2, b"\0\0\0\x05")], count=2))
        assert (facts.title_id, facts.content_type) == ("NPUB30000", None)

    def test_more_packets_than_the_bound_are_not_read(self):
        facts = read(pkg([], count=ps3_title.PKG_META_PACKETS_LIMIT + 1))
        assert facts.content_type is None

    def test_metadata_belongs_to_a_package_alone(self):
        with pytest.raises(ValueError):
            ps3_title.Ps3TitleFacts(ps3_title.SOURCE_PARAM_SFO, "BLUS30443", content_type=0)
        with pytest.raises(ValueError):
            ps3_title.Ps3TitleFacts(ps3_title.SOURCE_PKG, "NPUB30000", install_dir="NPUB30001")

    def test_the_real_machine_reads_it(self, tmp_path: Path):
        path = tmp_path / "Game.pkg"
        path.write_bytes(pkg([(0x2, b"\0\0\0\x05")]))
        result = RealMachine().read_ps3_title(str(path))
        assert result.status == PS3_TITLE_OK and result.facts is not None and result.facts.content_type == 5

    @pytest.mark.parametrize("spec", [{"content_type": "5"}, {"content_type": True}, {"install_dir": "X"}])
    def test_a_fixture_states_metadata_as_the_reader_does(self, spec):
        with pytest.raises(ValueError):
            FixtureMachine(
                {"/g.pkg": ""}, ps3_titles={"/g.pkg": {"source": "pkg", "title_id": "NPUB30000", **spec}}
            )


class _View:
    """The frontend's view, every probe answered from the files it is given; *unknown* answers neither way."""

    def __init__(self, files: set[str], dirs: set[str], unknown: frozenset[str] = frozenset()) -> None:
        self._files, self._dirs, self._unknown = files, dirs, unknown

    def _probe(self, hit: bool, path: str) -> Probe:
        return PROBE_UNKNOWN if path in self._unknown else PROBE_HIT if hit else PROBE_MISS

    def is_directory(self, path: str) -> Probe:
        return self._probe(path in self._dirs, path)

    def regular_file(self, path: str) -> Probe:
        return self._probe(path in self._files, path)

    def listing(self, directory: str) -> tuple[str, ...] | None:
        if directory in self._unknown:
            return None
        return tuple(os.path.basename(p) for p in (*self._files, *self._dirs) if os.path.dirname(p) == directory)

    def host_path(self, path: str) -> str | None:
        return "/host" + path


def _titles(facts: dict[str, ps3_title.Ps3TitleFacts]):
    def read_title(host: str) -> Ps3TitleResult:
        found = facts.get(host.removeprefix("/host"))
        return Ps3TitleResult(PS3_TITLE_OK, found) if found is not None else Ps3TitleResult("unreadable")

    return read_title


def _pkg(content_type: int | None, install_dir: str | None = None) -> ps3_title.Ps3TitleFacts:
    return ps3_title.Ps3TitleFacts(ps3_title.SOURCE_PKG, "NPUB30000", content_type, install_dir)


def _read(files: dict[str, ps3_title.Ps3TitleFacts | None], *, drive: str | None = HDD0, unknown=frozenset()):
    view = _View(set(files), {GAMES, "/roms/ps3/Game/Sub"}, unknown)
    known = {path: facts for path, facts in files.items() if facts is not None}
    return package_install.read_packages(GAMES, view, _titles(known), drive)  # pyright: ignore[reportArgumentType]


class TestWhatTheInstallTakes:
    def test_a_folder_is_its_files_by_name_case_aside_licences_first_then_packages(self):
        reading = _read(
            {
                f"{GAMES}/b.PKG": _pkg(5),
                f"{GAMES}/A.pkg": _pkg(0x0B),
                f"{GAMES}/key.rap": None,
                f"{GAMES}/key.edat": None,
                f"{GAMES}/.hidden.pkg": _pkg(5),
                f"{GAMES}/notes.txt": None,
            }
        )
        assert [t.package for t in reading.targets] == [f"{GAMES}/A.pkg", f"{GAMES}/b.PKG"]
        assert reading.licence_files == (f"{GAMES}/key.rap", f"{GAMES}/key.edat")
        assert reading.caveats == ()

    def test_a_game_goes_into_its_titles_folder_with_its_boot_file(self):
        (target,) = _read({f"{GAMES}/g.pkg": _pkg(5)}).targets
        assert target.destination == package_install.DESTINATION_GAME_FOLDER
        assert target.install_folder == package_install.PackageInstallPath(
            "/emu/dev_hdd0/game/NPUB30000", "/host/emu/dev_hdd0/game/NPUB30000"
        )
        assert target.boot_file is not None
        assert target.boot_file.frontend_path == "/emu/dev_hdd0/game/NPUB30000/USRDIR/EBOOT.BIN"

    def test_an_install_directory_replaces_the_title(self):
        (target,) = _read({f"{GAMES}/g.pkg": _pkg(4, "NPUB30001")}).targets
        assert target.install_folder is not None
        assert target.install_folder.frontend_path == "/emu/dev_hdd0/game/NPUB30001"

    def test_a_licence_goes_into_the_installers_exdata_and_boots_nothing(self):
        (target,) = _read({f"{GAMES}/g.pkg": _pkg(0x0B)}).targets
        assert target.destination == package_install.DESTINATION_LICENCE_FOLDER
        assert target.install_folder is not None and target.boot_file is None
        assert target.install_folder.frontend_path == "/emu/dev_hdd0/home/00000001/exdata"

    def test_a_theme_goes_elsewhere(self):
        (target,) = _read({f"{GAMES}/g.pkg": _pkg(0x09)}).targets
        assert (target.destination, target.install_folder) == (package_install.DESTINATION_ELSEWHERE, None)

    @pytest.mark.parametrize("name", ["../X", "", "A B"])
    def test_a_folder_name_rpcs3_refuses_or_escapes_is_not_followed(self, name):
        reading = _read({f"{GAMES}/g.pkg": _pkg(5, name)})
        assert reading.targets[0].destination == package_install.DESTINATION_UNREAD
        assert [c.code for c in reading.caveats] == [package_install.CAVEAT_PACKAGE_HEADER_UNREAD]

    def test_an_unread_header_states_no_destination(self):
        reading = _read({f"{GAMES}/g.pkg": None})
        assert reading.targets[0].destination == package_install.DESTINATION_UNREAD
        assert [c.code for c in reading.caveats] == [package_install.CAVEAT_PACKAGE_HEADER_UNREAD]

    def test_a_folder_holding_no_package_takes_nothing(self):
        reading = _read({f"{GAMES}/notes.txt": None})
        assert [c.code for c in reading.caveats] == [package_install.CAVEAT_PACKAGE_INSTALL_TAKES_NOTHING]

    def test_a_file_that_cannot_be_told_leaves_the_packages_unstated(self):
        reading = _read({f"{GAMES}/g.pkg": _pkg(5)}, unknown=frozenset({f"{GAMES}/g.pkg"}))
        assert reading.targets == ()
        assert [c.code for c in reading.caveats] == [package_install.CAVEAT_PACKAGE_INSTALL_PATH_UNREAD]

    def test_without_the_drive_no_folder_is_stated(self):
        reading = _read({f"{GAMES}/g.pkg": _pkg(5)}, drive=None)
        assert reading.targets[0].install_folder is None and reading.licence_folder is None
        assert [c.code for c in reading.caveats] == [package_install.CAVEAT_PACKAGE_DRIVE_UNRESOLVED]

    def test_the_arguments_are_the_headless_installers(self):
        assert package_install.install_arguments("/p") == ("--headless", "--installpkg", "/p")


class TestTheLicenceAccount:
    @pytest.mark.parametrize("accounts", [(), ("00000001",)])
    def test_the_installers_account_alone_needs_no_caveat(self, accounts):
        assert package_install.licence_account_caveat(accounts) is None

    @pytest.mark.parametrize("accounts", [None, ("00000002",), ("00000001", "00000002")])
    def test_another_account_or_an_unknown_listing_is_stated(self, accounts):
        caveat = package_install.licence_account_caveat(accounts)
        assert caveat is not None and caveat.code == package_install.CAVEAT_PACKAGE_LICENCE_ACCOUNT


class TestTheAnswer:
    def test_what_an_install_leaves_stands_only_beside_a_command(self):
        with pytest.raises(ValueError):
            atlas.PackageInstallAnswer(licence_files=("/x.rap",))
        with pytest.raises(ValueError):
            atlas.PackageInstallAnswer(exit_status_reports_outcome=False)

    def test_a_relative_path_is_refused(self):
        with pytest.raises(ValueError):
            atlas.EveryInstallation(()).package_install_command("ps3", "relative.pkg")
