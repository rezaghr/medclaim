"""Tiny deterministic PNG plots without an optional plotting dependency."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def _write_png(path: Path, pixels: list[list[tuple[int, int, int]]]) -> None:
    height = len(pixels)
    width = len(pixels[0])
    raw = b"".join(b"\x00" + bytes(channel for pixel in row for channel in pixel) for row in pixels)
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _chunk(b"IDAT", zlib.compress(raw, 9))
    payload += _chunk(b"IEND", b"")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def write_line_plot(
    path: Path,
    points: list[tuple[float, float]],
    *,
    diagonal: bool = False,
) -> None:
    width, height, margin = 640, 400, 40
    pixels = [[(255, 255, 255) for _ in range(width)] for _ in range(height)]

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            pixels[y][x] = color

    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        steps = max(abs(x1 - x0), abs(y1 - y0), 1)
        for step in range(steps + 1):
            x = round(x0 + (x1 - x0) * step / steps)
            y = round(y0 + (y1 - y0) * step / steps)
            for offset in (-1, 0, 1):
                set_pixel(x, y + offset, color)

    left, right = margin, width - margin
    top, bottom = margin, height - margin
    line(left, bottom, right, bottom, (40, 40, 40))
    line(left, bottom, left, top, (40, 40, 40))
    if diagonal:
        line(left, bottom, right, top, (170, 170, 170))
    ordered = sorted((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))) for x, y in points)
    coordinates = [
        (round(left + x * (right - left)), round(bottom - y * (bottom - top)))
        for x, y in ordered
    ]
    for first, second in zip(coordinates, coordinates[1:]):
        line(*first, *second, (31, 119, 180))
    for x, y in coordinates:
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                if dx * dx + dy * dy <= 9:
                    set_pixel(x + dx, y + dy, (214, 39, 40))
    _write_png(path, pixels)


def write_bar_plot(path: Path, values: list[float]) -> None:
    count = max(len(values), 1)
    points: list[tuple[float, float]] = []
    for index, value in enumerate(values):
        x = (index + 0.5) / count
        points.extend([(x - 0.3 / count, max(0.0, min(1.0, value))), (x + 0.3 / count, max(0.0, min(1.0, value)))])
    write_line_plot(path, points)
