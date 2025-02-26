import pytest

from odf.sbe import channels


@pytest.mark.parametrize(
    "channel,expected",
    [
        (0, (0, 1, 1)),
        (1, (1, 2, 0)),
        (2, (3, 4, 1)),
        (3, (4, 5, 0)),
        (4, (6, 7, 1)),
        (5, (7, 8, 0)),
        (6, (9, 10, 1)),
        (7, (10, 11, 0)),
    ],
)
def test_get_volt_indicies(channel, expected):
    assert channels.get_volt_indicies(channel) == expected
