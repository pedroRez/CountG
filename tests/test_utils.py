from utils.contagem_video import (
    get_line_and_direction_config,
    LINE_VERTICAL,
    MOVE_LR,
    is_crossing_diagonal_line,
    MOVE_TL_BR,
)


def test_get_line_and_direction_config_east():
    line_type, direction, line_points, pos, _ = get_line_and_direction_config(
        'E', width=100, height=50
    )
    assert line_type == LINE_VERTICAL
    assert direction == MOVE_LR
    assert line_points == ((50, 0), (50, 50))
    assert pos == 50


def test_is_crossing_diagonal_line():
    p_prev = (10, 20)
    p_curr = (30, 25)
    line_p1 = (0, 0)
    line_p2 = (100, 100)
    crossed = is_crossing_diagonal_line(
        p_prev[0], p_prev[1],
        p_curr[0], p_curr[1],
        line_p1, line_p2, MOVE_TL_BR,
    )
    assert crossed is True
