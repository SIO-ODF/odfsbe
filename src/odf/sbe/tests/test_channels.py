import numpy as np
import pytest
import xarray as xr

from odf.sbe import channels

#   Create a testing line (RR2204, 00301 line 40,000). From xmlcon:
#   5xF (15 bytes) [0:15]
#   8xV (12 bytes) [15:27]
#   Metadata:
#       NMEA la/lon (7 bytes) [27:34]
#       SBE9 core (3 bytes) [34:37]
#       System time "scan" (4 bytes) [37:41]
# fmt: off
sample_line = np.array(
    [
        [
            20,  77,  30,  27, 142,  27, 129, 180, 180,  18, 153, 136,  23, 156, 189,
            253, 225,  86, 185,  26,  68,   0,  95, 245, 130, 111, 255,
            24, 205,   5, 101, 126,  54,   0,
            144, 211, 184,
            193, 207, 114,  98
        ]
    ],
)
# fmt: on


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
        (6, 1, 4.993894993894994),  # 1 f_s moves back 3 bytes to v4
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


def test_nmeaposition():
    bytes_in = xr.DataArray(sample_line[:, 27:34])
    result = channels._nmeaposition(bytes_in)

    assert all(key in result for key in ["latitude", "longitude", "newpos"])
    assert result["latitude"].item() == (24 << 16 | 205 << 8 | 5) / 50000
    assert result["longitude"].item() == (101 << 16 | 126 << 8 | 54) / 50000
    assert not result["newpos"].item()

    with pytest.raises(AttributeError) as err:
        channels._nmeaposition("a string")
    assert "'str' object has no attribute 'astype'" in str(err.value)


def test_sbe_time():
    bytes_in = xr.DataArray(sample_line[:, 37:41], dims=["scan", "bytes_per_scan"])

    result = channels._sbe_time(bytes_in, sbe_type="ScanTime")
    assert result.name == "ScanTime"
    epoch = np.datetime64("1970-01-01")
    byte_positions = np.array([1 << 0, 1 << 8, 1 << 16, 1 << 24], dtype=np.uint32)
    expected_timestamp = np.datetime64(
        epoch
        + np.timedelta64(
            (bytes_in.astype(np.uint32) * byte_positions).sum().item(), "s"
        )
    )

    assert np.datetime64(result.item()) == expected_timestamp

    result = channels._sbe_time(bytes_in, sbe_type="NmeaTime")
    assert result.name == "NmeaTime"
    epoch = np.datetime64("2000-01-01")
    expected_timestamp = np.datetime64(
        epoch
        + np.timedelta64(
            (bytes_in.astype(np.uint32) * byte_positions).sum().item(), "s"
        )
    )

    assert np.datetime64(result.item()) == expected_timestamp


def test_sbe9core():
    bytes_in = xr.DataArray(
        sample_line[:, 34:37], dims=["scan", "bytes_per_scan"]
    ).astype("uint8")
    result = channels._sbe9core(bytes_in)

    expected_ptemp = (144 << 8 | 211) >> 4
    expected_pump = bool(211 & 1)
    expected_switch = bool((211 >> 1) & 1)
    expected_sampler = bool((211 >> 2) & 1)
    expected_modem = bool((211 >> 3) & 1)
    expected_modulo = 184

    assert all(
        key in result for key in ["ptempC", "pump", "bct", "HBBotCls", "modem", "mod"]
    )
    assert result["ptempC"].item() == expected_ptemp
    assert result["pump"].item() == expected_pump
    assert result["bct"].item() == expected_switch
    assert result["HBBotCls"].item() == expected_sampler
    assert result["modem"].item() == expected_modem
    assert result["mod"].item() == expected_modulo


def test_metadata():
    hex_data = xr.DataArray(sample_line, dims=["scan", "bytes_per_scan"]).astype(
        "uint8"
    )
    cfg = {
        "FrequencyChannelsSuppressed": 0,
        "VoltageWordsSuppressed": 0,
        "SurfaceParVoltageAdded": 0,
        "NmeaPositionDataAdded": 1,
        "NmeaDepthDataAdded": 0,
        "NmeaTimeAdded": 0,
        "ScanTimeAdded": 1,
    }
    metadata = channels.get_metadata(hex_data, cfg)

    assert isinstance(metadata, dict)
    assert "ScanTime" in metadata
    assert np.array_equal(
        metadata["ScanTime"], np.array(["2022-05-04T19:10:57"], dtype="datetime64[s]")
    )
    assert metadata["latitude"] == 32.50698
    assert not metadata["newpos"]
    assert metadata["pump"]

    #   Line 0, before the pumps have turned on. Same cast, same config.
    # fmt: off
    hex_data = xr.DataArray(
        np.array([[
            19, 116, 115,  11,  67, 236, 129, 167, 153,  18,  60,  31,  10,
            32, 183, 254,  81,  12, 186,  58, 140,   1, 175, 246, 182, 239,
            255,  24, 205,   6, 101, 126,  49,   0, 145,  66, 120,  62, 201,
            114,  98
        ]], dtype=np.uint8),
        dims=["scan", "bytes_per_scan"]).astype("uint8")
    # fmt: on

    metadata = channels.get_metadata(hex_data, cfg)
    assert not metadata["pump"]
