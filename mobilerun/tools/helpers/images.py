"""Small image helpers used by screenshot-only device backends."""

from __future__ import annotations

import struct


def image_dimensions(image: bytes) -> tuple[int, int]:
    """Return ``(width, height)`` for PNG or JPEG bytes."""
    if image.startswith(b"\x89PNG\r\n\x1a\n") and len(image) >= 24:
        width, height = struct.unpack(">II", image[16:24])
        return int(width), int(height)

    if image.startswith(b"\xff\xd8"):
        return _jpeg_dimensions(image)

    raise ValueError("Unsupported screenshot image format. Expected PNG or JPEG.")


def _jpeg_dimensions(image: bytes) -> tuple[int, int]:
    offset = 2
    length = len(image)
    while offset + 9 < length:
        if image[offset] != 0xFF:
            offset += 1
            continue

        while offset < length and image[offset] == 0xFF:
            offset += 1
        if offset >= length:
            break

        marker = image[offset]
        offset += 1

        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA:
            break
        if offset + 2 > length:
            break

        segment_length = int.from_bytes(image[offset : offset + 2], "big")
        if segment_length < 2:
            raise ValueError("Invalid JPEG segment length.")

        if _is_start_of_frame(marker):
            if offset + 7 > length:
                break
            height = int.from_bytes(image[offset + 3 : offset + 5], "big")
            width = int.from_bytes(image[offset + 5 : offset + 7], "big")
            return int(width), int(height)

        offset += segment_length

    raise ValueError("Could not read JPEG dimensions.")


def _is_start_of_frame(marker: int) -> bool:
    return marker in {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
