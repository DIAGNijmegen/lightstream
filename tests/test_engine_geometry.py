import pytest

from lightstream.core.engine.geometry import aligned_step, full_output_size, iter_tiles, output_window, tile_grid


def test_tile_grid_clamps_last_tiles_and_marks_sides():
    assert tile_grid(10, 11, 6, 6, 4, 4) == (2, 3)
    tiles = list(iter_tiles(10, 11, 6, 6, 4, 4))
    assert [(y, x) for y, x, _ in tiles] == [(0, 0), (0, 4), (0, 5), (4, 0), (4, 4), (4, 5)]
    assert tiles[0][2].top and tiles[0][2].left
    assert tiles[-1][2].bottom and tiles[-1][2].right


def test_alignment_and_output_windows():
    assert aligned_step([31, 28], [4, 6]) == 24
    assert full_output_size(18, 10, 6, 2) == 10
    assert output_window(8, 2, 10, 6, 1, 1, False, True) == (5, 10, 1, 6)


def test_invalid_step_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        tile_grid(10, 10, 5, 5, 0, 2)
