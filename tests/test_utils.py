"""Testes para utilidades de contagem de vídeo.

O módulo valida a configuração de linhas e direções e a detecção de
cruzamentos de linhas diagonais.

Tests for video counting utilities.

The module validates line and direction configuration and detection of
diagonal line crossings.
"""

import subprocess
from types import SimpleNamespace

import cv2
import pytest

from utils.contagem_video import (  # isort: skip
    LINE_VERTICAL,
    MOVE_LR,
    MOVE_TL_BR,
    apply_rotation,
    get_line_and_direction_config,
    get_video_rotation,
    is_crossing_diagonal_line,
)


def test_get_line_and_direction_config_east():
    """Return a centered vertical line moving from left to right for east."""

    line_type, direction, line_points, pos, _ = get_line_and_direction_config(
        "E", width=100, height=50
    )
    assert line_type == LINE_VERTICAL
    assert direction == MOVE_LR
    assert line_points == ((50, 0), (50, 50))
    assert pos == 50


def test_get_line_and_direction_config_invalid_orientation():
    """Raise ValueError for unsupported orientation codes."""

    with pytest.raises(ValueError):
        get_line_and_direction_config("INVALID", width=100, height=50)


def test_is_crossing_diagonal_line():
    """Detect crossing of a top-left to bottom-right diagonal line."""

    p_prev = (10, 20)
    p_curr = (30, 25)
    line_p1 = (0, 0)
    line_p2 = (100, 100)
    crossed = is_crossing_diagonal_line(
        p_prev[0],
        p_prev[1],
        p_curr[0],
        p_curr[1],
        line_p1,
        line_p2,
        MOVE_TL_BR,
    )
    assert crossed is True


def test_rotation_metadata_and_application(tmp_path, monkeypatch):
    """Rotate frame based on metadata / Rotaciona frame segundo metadados."""

    video_file = tmp_path / "rot90.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=40x20:rate=1",
            "-t",
            "1",
            str(video_file),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    def fake_run(cmd, stdout, stderr, text, check):
        return SimpleNamespace(stdout="90")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rotation = get_video_rotation(str(video_file))
    assert rotation == 90

    frame = [[0] * 40 for _ in range(20)]

    def fake_rotate(f, code):
        if code == cv2.ROTATE_90_CLOCKWISE:
            return [list(row) for row in zip(*f[::-1])]
        if code == cv2.ROTATE_180:
            return [row[::-1] for row in f[::-1]]
        if code == cv2.ROTATE_90_COUNTERCLOCKWISE:
            return [list(row) for row in zip(*f)][::-1]
        return f

    monkeypatch.setattr(cv2, "ROTATE_90_CLOCKWISE", 0, raising=False)
    monkeypatch.setattr(cv2, "ROTATE_180", 1, raising=False)
    monkeypatch.setattr(cv2, "ROTATE_90_COUNTERCLOCKWISE", 2, raising=False)
    monkeypatch.setattr(cv2, "rotate", fake_rotate, raising=False)

    rotated = apply_rotation(frame, rotation)
    assert len(rotated) == 40 and len(rotated[0]) == 20
