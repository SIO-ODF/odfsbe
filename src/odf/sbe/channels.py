"""
Module for handling "channel" data columns within the SeaBird hex.
"""

import datetime

import numpy as np
import pandas as pd
import xarray as xr


def get_volt_indicies(n):
    """
    Calculate the HEX indices of a given voltage channel
    """
    start = n // 2 * 3
    high = n % 2
    return start + high, start + high + 1, 1 - high


def get_voltage(hex, channel, freq_supressed):
    """
    Compute voltage for given voltage channel
    """
    offset = (5 - freq_supressed) * 3

    first_byte_idx, second_byte_idx, shift = get_volt_indicies(channel)
    first_byte_idx += offset
    second_byte_idx += offset

    data = hex[:, first_byte_idx].astype("uint16") << 8
    data = data | hex[:, second_byte_idx]
    data = data >> (4 * shift)
    data = data & 4095
    return 5 * (1 - (data / 4095))


def get_frequency(hex, channel):
    """
    Compute frequency for given frequency channel
    """
    m = 3 * channel
    data = hex[:, m].astype("uint32") << 8
    data = (data | hex[:, m + 1]) << 8
    data = data | hex[:, m + 2]
    return data / 256


def _sbe_time(bytes_in, sbe_type="scan", reverse=False):
    """
    Determine UTC datetime from SBE format of 4 bytes, using either scan or NMEA.

    See legacy code sbe_reader.SBEReader._sbe_time and _reverse_bytes for
    historical examples when ODF worked on metadata as strings.

    Inputs:
    bytes_in: xarray DataArray of 4xn bytes.
    sbe_type: string of type of time to pull from ("nmea" or "scan")
    reverse: A historical argument for reversing byte order.

    Output:
    xarray 1xn DataArray of float64 of variable "Timestamp" in UTC.
    Check these values using
    `pd.to_datetime(data_to_write.values.flatten(), utc=True)`
    """
    if bytes_in.sizes["bytes_per_scan"] != 4:
        raise ValueError("Each scan should contain exactly 4 bytes.")

    if reverse:
        # When working with hex strings, both "scan" and "nmea" required flipping the bytes
        # TODO: Do we need this? Gave me bad conversions when reversed
        bytes_out = bytes_in[:, ::-1]
    else:
        bytes_out = bytes_in

    # Define epoch start times based on the sbe conversion type
    if sbe_type == "scan":
        time_start = datetime.datetime(1970, 1, 1, tzinfo=datetime.UTC)
    elif sbe_type == "nmea":
        time_start = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)

    # Convert bytes to integer timestamps with weights modifier
    weights = np.array([1, 256, 65536, 16777216], dtype=np.uint32)
    all_seconds = (bytes_out.astype(np.uint32) * weights).sum(dim="bytes_per_scan")

    # Something human-readable
    timestamps = pd.to_datetime(
        all_seconds.values + time_start.timestamp(), unit="s", utc=True
    ).values.reshape(-1, 1)
    timestamps = timestamps.astype(np.float64)

    return xr.DataArray(
        timestamps,  # 2D array
        dims=["scan", "variable"],
        coords={
            "scan": bytes_in.coords[
                "scan"
            ],  # Keep the scan dimension from the input data
            "variable": ["Timestamp"],  # Only one variable: "Timestamp"
        },
    )


def _nmeaposition(bytes_in):
    """
    Determine location from SBE format of 8-bit integers.

    Requires bytes_in of 7 bytes in a DataArray or array-like format.

    See legacy code sbe_reader.SBEReader._location_fix context.

    Input:
    b1, b2, b3: three components of Latitude
    b4, b5, b6: three components of Longitude
    b7: sign for lat/lon, is used for sign corrections

    Output:
    DataArray of:
    latitude, longitude, new fix
    Latitude is a float
    Longitude is a float
    If it is a "new fix" it will be true, otherwise false

    For byte 7:
    If bit 1 in byte_pos is 1, this is a new position
    If bit 8 in byte_pos is 1, lat is negative
    If bit 7 in byte_pos is 1, lon is negative
    """
    b1 = bytes_in[:, 0]
    b2 = bytes_in[:, 1]
    b3 = bytes_in[:, 2]
    b4 = bytes_in[:, 3]
    b5 = bytes_in[:, 4]
    b6 = bytes_in[:, 5]
    b7 = bytes_in[:, 6]

    #   Compute latitude and longitude
    lat = (b1 * 65536 + b2 * 256 + b3) / 50000
    lon = (b4 * 65536 + b5 * 256 + b6) / 50000

    #   Hex masks to extract specific bits of byte 7
    mask_lat_pos = 0x80  #   Bit 8 10000000, if lat negative
    mask_lon_pos = 0x40  #   Bit 7 01000000, if lon negative
    mask_new_fix = 0x01  #   Bit 1 00000001, if "new position"

    #   Apply byte 7 modifications
    lat = np.where(b7 & mask_lat_pos, -lat, lat)
    lon = np.where(b7 & mask_lon_pos, -lon, lon)
    flag_new_fix = (b7 & mask_new_fix).astype(bool)

    output = xr.DataArray(
        np.column_stack((lat, lon, flag_new_fix)),
        dims=["scan", "variable"],
        coords={"variable": ["lat", "lon", "sign_fix"]},
    )

    return output


def _sbe9core(bytes_in):
    """
    Handle the bundle of SBE9 core metadata columns: SBE9 temperature, pump status, contact switch, bottle fire, modem, modulo.
    """

    #   SBE9 temp: 12-bit number if binary notation of temperature from 0-4095
    temp = bytes_in[:, 0].astype("uint16") << 8
    temp |= bytes_in[:, 1]
    temp = temp >> 4

    #   CTD status
    pump = bytes_in[:, 1] & 1  #   Gets shifted
    switch = bytes_in[:, 1] >> 1 & 1
    sampler = bytes_in[:, 1] >> 2 & 1
    modem = bytes_in[:, 1] >> 3 & 1

    #   Modulo byte
    modulo = bytes_in[:, 2]

    return xr.DataArray(
        np.column_stack((temp, pump, switch, sampler, modem, modulo)),
        dims=["scan", "variable"],
        coords={
            "variable": [
                "sbe9_temp",
                "pump_status",
                "contact_switch",
                "bottle_fire",
                "modem_sensed",
                "modulo_error",
            ]
        },
    )


def metadata_wrapper(hex_data, cfg):
    """
    A wrapper of sorts to handle columnar metadata in the source HEX file.

    Columns of the HEX are output in a specific order, based on configuration:
    ---done with `hex_to_f`, `hex_to_v`---
    1) Data from the instrument (written to netCDF as `engineering`)
        a) Frequency (3 bytes each)
        b) Voltage (12 bits each)
    ---this wrapper---
    2) Surface Par (3 bytes) (ODF historically has limited support for this)
    3) NMEA lat/lon (7 bytes)
    4) NMEA depth (3 bytes)
    5) NMEA time (4 bytes) (low byte first)
    6) Additional Data from the instrument
        a) Pressure temp (12 bits)
        b) pump status (4 bits)
        c) modulo byte (1 byte)
    7) System time (4 bytes) (low byte first)

    Information about these steps (bytes, equations) is available in SBE's
    SBEDataProcessing and SeaSave manuals.

    Inputs:
    hex_data: xarray DataArray of the entire hex file
    cfg: the `data.xmlcon_box` from the XMLCON file, indicating which features
    are added to the HEX rows
    """

    #   Start by figuring out the starting byte for the metadata
    f_s, v_s = (
        cfg[i] for i in ["FrequencyChannelsSuppressed", "VoltageWordsSuppressed"]
    )
    num_f = 5 - f_s  #   3 bytes per f
    num_v = 8 - v_s  #   3 bytes per 2 v
    start_byte_ix = int(num_f * 3 + num_v * 1.5)

    #   Track where we are in the metadata columns
    ix_tracker = start_byte_ix

    # Initialize an xarray Dataset
    meta_out = xr.Dataset()
    meta_out["scan_number"] = xr.DataArray(
        np.arange(1, len(hex_data) + 1), dims=["scan"]
    )

    # Metadata is in a specific order. If it's there, extract specific sizes
    # and increment the current column index. If it isn't, don't increment.

    # surface_par_voltage
    if cfg["SurfaceParVoltageAdded"]:
        #   handle the surface PAR - TODO: historically ODF hasn't done this?
        # Manual P39: Surface PAR: 12-bit number (N) is binary notation of analog voltage. N =
        # 4095 for 0 V, 0 for 5 V.
        # V = N ÷ 819
        # e.g.: byte 34 = 11110011 byte 35 = 01110100
        # N = 001101110100 = 884 decimal; V = 884 ÷ 819 = 1.079 V
        col_extracts = hex_data[:, ix_tracker : ix_tracker + 3]
        ix_tracker = ix_tracker + 3

    # nmea_position_data
    if cfg["NmeaPositionDataAdded"]:
        col_extracts = hex_data[:, ix_tracker : ix_tracker + 7].astype(int)
        data_to_write = _nmeaposition(col_extracts)
        ix_tracker += 7
        #   Unpack to variable in the dataset
        for var_name, values in zip(
            data_to_write.coords["variable"].values, data_to_write.T.values, strict=True
        ):
            meta_out[var_name] = xr.DataArray(values, dims=["scan"])

    # nmea_depth_data
    if cfg["NmeaDepthDataAdded"]:
        col_extracts = hex_data[:, ix_tracker : ix_tracker + 3]
        ix_tracker = ix_tracker + 3

    # nmea_time
    if cfg["NmeaTimeAdded"]:
        col_extracts = hex_data[:, ix_tracker : ix_tracker + 4]
        data_to_write = _sbe_time(col_extracts, "nmea")
        ix_tracker += 4
        meta_out["nmea_timestamp"] = xr.DataArray(
            data_to_write.values[:, 0], dims=["scan"]
        )

    #   The following 3 are always true for SBE9:
    #   * pressure_temp - 1.5 bytes
    #   * flag_ctd_status (pump and bottle fire) - 0.5 bytes
    #   * modulo errors - 1 byte
    col_extracts = hex_data[:, ix_tracker : ix_tracker + 3]
    data_to_write = _sbe9core(col_extracts)
    ix_tracker += 3
    for var_name, values in zip(
        data_to_write.coords["variable"].values, data_to_write.T.values, strict=True
    ):
        meta_out[var_name] = xr.DataArray(values, dims=["scan"])

    # scan_time
    if cfg["ScanTimeAdded"]:
        col_extracts = hex_data[:, ix_tracker : ix_tracker + 4]
        data_to_write = _sbe_time(col_extracts, "scan")
        ix_tracker += 4
        meta_out["scan_timestamp"] = xr.DataArray(
            data_to_write.values[:, 0], dims=["scan"]
        )
    else:
        print("No ScanTimeAdded in XMLCON - building manually from frequency.")
        #   And then we do what sbeReader used to do

    if ix_tracker > len(hex_data[0]):
        print(
            f"Uh oh, might not have incremented metadata correctly."
            f" Got {ix_tracker} bytes out of a possible {len(hex_data[0])}."
        )

    return meta_out
