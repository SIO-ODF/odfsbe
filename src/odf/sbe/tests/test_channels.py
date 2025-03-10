import pytest

from odf.sbe import channels
import numpy as np

sample_line = np.array([[
    20,  77,  30,  27, 142,  27, 129, 180, 180,  18, 153, 136,  23,
    156, 189, 253, 225,  86, 185,  26,  68,   0,  95, 245, 130, 111,
    255,  24, 205,   5, 101, 126,  54,   0, 144, 211, 184, 193, 207,
    114,  98
]])

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


@pytest.mark.parametrize(
    "channel,f_s,expected",
    [
        (0, 0, 0.04029304029304015),
        (1, 0, 4.582417582417582),
        (2, 0, 1.3846153846153846),
        (3, 0, 1.791208791208791),
        (4, 0, 4.993894993894994),
        (5, 0, 0.012210012210012167),
        (6, 0, 2.452991452991453),
        (7, 0, 0.0),
        (6, 1, 4.993894993894994),   # 1 f_s moves back 3 bytes to v4
    ],
)    
def test_get_voltage(channel, f_s, expected):
    assert channels.get_voltage(sample_line, channel, f_s).item() == expected
    with pytest.raises(TypeError) as err:
        channels.get_voltage(sample_line, "6", 0)
    assert "unsupported operand type(s) for //: 'str' and 'int'" in str(err.value)

@pytest.mark.parametrize(
    "channel, expected",
    [
        (0, (20 * (256**2) + 77 * 256 + 30) / 256),
        (1, (27 * (256**2) + 142 * 256 + 27) / 256),
        (2, (129 * (256**2) + 180 * 256 + 180) / 256),
        (3, (18 * (256**2) + 153 * 256 + 136) / 256),
        (4, (23 * (256**2) + 156 * 256 + 189) / 256),
    ],
)
def test_get_frequency(channel, expected):
    assert channels.get_frequency(sample_line, channel)[0] == expected

    with pytest.raises(TypeError) as err:
        channels.get_frequency("not an array", 0)
    assert "string indices must be integers, not 'tuple'" in str(err.value)
    with pytest.raises(IndexError) as err:
        channels.get_frequency(sample_line, "0")
    assert "only integers, slices (`:`)" in str(err.value)
