"""Write small GameCube and Wii disc images for the reader's tests — built from the container layouts.

The writer is the reader's mirror and nothing more: a disc header with a
game id and the platform's magic, a GameCube region code at 0x458 or a Wii
partition table at 0x40000 pointing at one partition whose first bytes are a
v0 ticket, and that disc wrapped in each container :mod:`atlas.dolphin_disc`
opens — plain, CISO, GCZ, WBFS, WIA and RVZ. Nothing here is a real disc.
"""

from __future__ import annotations

import hashlib
import struct
import zlib

GC_MAGIC = 0xC2339F3D
WII_MAGIC = 0x5D1C9EA3
PARTITION = 0x50000
TICKET_SIZE = 0x2A4
RSA2048 = 0x00010001


def gc_disc(game_id: bytes = b"GALE01", *, region: int = 1, revision: int = 0, size: int = 0x800) -> bytes:
    """A GameCube disc's first *size* bytes: the id, the revision, the magic at 0x1C and the region code."""
    disc = bytearray(size)
    disc[0:6] = game_id
    disc[7] = revision
    struct.pack_into(">I", disc, 0x1C, GC_MAGIC)
    if size >= 0x45C:
        struct.pack_into(">I", disc, 0x458, region)
    return bytes(disc)


def wii_disc(
    game_id: bytes = b"RSBE01",
    *,
    title_id: int = 0x0001000052534245,
    signature: int = RSA2048,
    partition_type: int = 0,
    groups: tuple[int, ...] = (0,),
) -> bytes:
    """A Wii disc through its one partition's ticket: the magic at 0x18, the table at 0x40000, the ticket."""
    disc = bytearray(PARTITION + TICKET_SIZE)
    disc[0:6] = game_id
    struct.pack_into(">I", disc, 0x18, WII_MAGIC)
    table = 0x40020
    for group in groups:
        struct.pack_into(">II", disc, 0x40000 + group * 8, 1, table >> 2)
    struct.pack_into(">II", disc, table, PARTITION >> 2, partition_type)
    struct.pack_into(">I", disc, PARTITION, signature)
    struct.pack_into(">Q", disc, PARTITION + 0x1DC, title_id)
    return bytes(disc)


def ciso(disc: bytes, *, block_size: int = 0x8000) -> bytes:
    """*disc* as a CISO: every block stored, in order, behind the 0x8000-byte header."""
    blocks = [disc[i : i + block_size].ljust(block_size, b"\0") for i in range(0, len(disc), block_size)]
    header = bytearray(0x8000)
    header[0:4] = b"CISO"
    struct.pack_into("<I", header, 4, block_size)
    for index in range(len(blocks)):
        header[8 + index] = 1
    return bytes(header) + b"".join(blocks)


def gcz(disc: bytes, *, block_size: int = 0x8000, raw: tuple[int, ...] = ()) -> bytes:
    """*disc* as a GCZ: each block one zlib stream, or stored raw with the pointer's top bit where *raw* names it."""
    blocks = [disc[i : i + block_size].ljust(block_size, b"\0") for i in range(0, len(disc), block_size)]
    stored = [block if index in raw else zlib.compress(block) for index, block in enumerate(blocks)]
    pointers, offset = [], 0
    for index, data in enumerate(stored):
        pointers.append(offset | (1 << 63 if index in raw else 0))
        offset += len(data)
    header = struct.pack("<IIQQII", 0xB10BC001, 0, offset, len(blocks) * block_size, block_size, len(blocks))
    table = b"".join(struct.pack("<Q", p) for p in pointers)
    hashes = b"".join(struct.pack("<I", zlib.adler32(data)) for data in stored)
    return header + table + hashes + b"".join(stored)


def wbfs(disc: bytes, *, hd_shift: int = 9, wbfs_shift: int = 20) -> bytes:
    """*disc* as a one-disc WBFS whose first disc cluster is stored at WBFS sector 1."""
    hd, sector = 1 << hd_shift, 1 << wbfs_shift
    assert len(disc) <= sector
    image = bytearray(sector + len(disc))
    image[0:4] = b"WBFS"
    struct.pack_into(">IBB", image, 4, -(-len(image) // hd), hd_shift, wbfs_shift)
    image[12] = 1
    image[hd : hd + 256] = disc[:256]
    struct.pack_into(">H", image, hd + 256, 1)
    image[sector:] = disc
    return bytes(image)


def wia(disc: bytes, *, rvz: bool = False, version: int = 0x01000000) -> bytes:
    """*disc*'s first 0x80 bytes in a WIA or RVZ second header, both headers' hashes closing them."""
    header_2 = bytearray(0xDC)
    header_2[0x10:0x90] = disc[:0x80].ljust(0x80, b"\0")
    size = 0x48 + len(header_2)
    head = struct.pack(
        ">4sIII20sQQ",
        b"RVZ\x01" if rvz else b"WIA\x01",
        version,
        version,
        len(header_2),
        hashlib.sha1(header_2).digest(),
        len(disc),
        size,
    )
    return head + hashlib.sha1(head).digest() + bytes(header_2)
