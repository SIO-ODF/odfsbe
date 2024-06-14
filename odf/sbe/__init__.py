from typing import Literal, Generator
from pathlib import Path

import numpy as np
import xarray as xr

ERRORS = Literal["store", "raise", "ignore"]

def hex_to_dataset(hex:str, errors: ERRORS="raise") -> xr.Dataset:
    _comments = []  #   hex header comments written by deck box/SeaSave
    out = []        #   hex bytes out

    datalen = 0
    for lineno, line in enumerate(hex.splitlines(), start=1):
        if "number of bytes per scan" in line.lower():
            datalen = int(line.split("= ")[1])
            linelen = datalen * 2

        if line.startswith("*"): # comment
            _comments.append(line)
            continue

        if datalen == 0:
            raise ValueError(f"Could not find number of bytes per scan in {lineno} lines")

        if len(line) != linelen:
            if errors == "raise":
                raise ValueError(f"invalid scan lengths line: {lineno}")
            elif errors == "ignore":
                continue
            elif errors == "store":
                raise NotImplementedError("better figure out how to do this")

        out.append([*bytes.fromhex(line)])
    header = "\n".join(_comments)
    data = np.array(out, dtype=np.uint8)

    data_array = xr.DataArray(data, dims=["scan","bytes_per_scan"])
    data_array.encoding["zlib"] = True  # compress the data
    data_array.encoding["complevel"] = 6  # use compression level 6
    data_array.encoding["chunksizes"] = (60*60*24, 1) # chunk every hour of data (for 24hz data), and each column seperately
    # This is about 3~4mb chunks uncompressed depending on how many channels there are

    return xr.Dataset({
        "hex": data_array  # TODO: decide on the name of this variable as we will live with it "forever", "hex" is awful, but maybe the best?
    },
    attrs={
        "hex_header": header
    })

def read_hex(path, xmlcon=(".XMLCON", ".xmlcon"), bl=(".bl", ), hdr=(".hdr", )) -> xr.Dataset:
    hex_path = Path(path)
    xmlcon_path = None
    bl_path = None
    hdr_path = None

    if xmlcon is not None:
        for suffix in xmlcon:
            if (xmlcon_p := hex_path.with_suffix(suffix)).exists():
                xmlcon_path = xmlcon_p

    if bl is not None:
        for suffix in bl:
            if (bl_p := hex_path.with_suffix(suffix)).exists():
                bl_path = bl_p
    if hdr is not None:
        for suffix in hdr:
            if (hdr_p := hex_path.with_suffix(suffix)).exists():
                hdr_path = hdr_p

    ds = hex_to_dataset(hex_path.read_text(), errors="ignore")
    ds.attrs["hex_filename"] = hex_path.name
    if xmlcon_path is not None:
        ds.attrs["xmlcon_filename"] = xmlcon_path.name
        ds.attrs["xmlcon"] = xmlcon_path.read_text()
    if bl_path is not None:
        ds.attrs["bl_filename"] = bl_path.name
        ds.attrs["bl"] = bl_path.read_text()

    if hdr_path is not None:
        ds.attrs["hdr_filename"] = hdr_path.name
        # TODO... figure out why this file doesn't seem to be matching the hex header (line endings?), and what to do when it doesnt

    return ds

def ds_to_hex(ds: xr.Dataset) -> Generator[bytes, None, None]:
    """The input for this function is the output of the above (we don't have a spec yet).

    When first made, this was meant to recreate the input byte for byte when moving the data across
    a very low bandwidth network (USAP).

    It works by dumping numpy to a python bytes object, dumping that to hex, uppercasing and encoding 
    using UTF8 (ASCII in this case). The output of this is a single long string which is then
    chunked though to split all the lines.
    """
    yield "\r\n".join(ds.comments.splitlines()).encode("utf8")
    yield b"\r\n"
    data = bytes(ds.hex.as_numpy().values).hex().upper().encode("utf8")
    row_len = ds.dims["bytes_per_scan"] * 2
    
    for row in range(ds.dims["scan"]):
        start = row * row_len
        stop = row * row_len + row_len
        yield data[start:stop]
        yield b"\r\n"