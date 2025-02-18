from typing import Literal, Generator
from pathlib import Path
from hashlib import md5

import numpy as np
import xarray as xr

ERRORS = Literal["store", "raise", "ignore"]

"""
filename to var mapping
*.hex -> "hex" - uint8 decoded hex data (for compression)
*.xmlcon -> "xmlcon" - stored char data
*.bl -> "bl" - stored char data
*.hdr -> "hdr" stored char data

If the hex has bad scans and errors is set to "store" (the default)
"hex_errors" -> string of the bad lines

hex has the first dim of scan, which is also a coordinate, this will be continiously incrimenting by 1 if nothing is wrong with the input hex

each variable will have following attrs:
"filename" - name of the input file
"Content-MD5" - md5 hash of the input file
"charset" -> Input text encoding for round trip
"_Encoding" -> Set by xarray always be utf8 for char/string vars, only in the actual netCDF file

The hex var gets some special attrs:
header - the part of the file that were not hex
"""

def hex_to_dataset(path:Path, errors: ERRORS="store", encoding="CP437", content_md5=True) -> xr.Dataset:
    _comments = []  #   hex header comments written by deck box/SeaSave
    out_idx = []    #   zero indexed "row" of the hex line, used for reconsturction of bad files
    out = []        #   hex bytes out
    hex = path.read_text(encoding)

    error_idx = []
    error_lines = []
    datalen = 0
    header_len = 0
    for lineno, line in enumerate(hex.splitlines(), start=1):
        if "number of bytes per scan" in line.lower():
            datalen = int(line.split("= ")[1])
            linelen = datalen * 2

        if line.startswith("*"): # comment
            _comments.append(line)
            header_len = lineno
            continue

        if datalen == 0:
            raise ValueError(f"Could not find number of bytes per scan in {lineno} lines")

        if len(line) != linelen:
            if errors == "raise":
                raise ValueError(f"invalid scan lengths line: {lineno}")
            elif errors == "ignore":
                continue
            elif errors == "store":
                error_idx.append(lineno - header_len)
                error_lines.append(line)
                continue

        out_idx.append(lineno - header_len)
        out.append([*bytes.fromhex(line)])
    header = "\n".join(_comments)
    data = np.array(out, dtype=np.uint8)

    data_array = xr.DataArray(data, dims=["scan","bytes_per_scan"], coords={"scan": out_idx})
    data_array.attrs["header"] = header  # utf8 needs to be encoded using .attrs["charset"] when written back out

    data_array.attrs["filename"] = path.name
    if content_md5:
        data_array.attrs["Content-MD5"] = md5(path.read_bytes()).hexdigest()
    data_array.attrs["charset"] = encoding

    # Encoding is instructions for xarray
    data_array.encoding["zlib"] = True  # compress the data
    data_array.encoding["complevel"] = 6  # use compression level 6
    data_array.encoding["chunksizes"] = (60*60*24, 1) # chunk every hour of data (for 24hz data), and each column seperately
    # This is about 3~4mb chunks uncompressed depending on how many channels there are
    data_ararys = {
        "hex": data_array
    }

    if errors == "store" and len(error_lines) > 0:
        # make a string array of the bad lines
        error_data_array = xr.DataArray(error_lines, dims=["scan_errors"], coords={"scan_errors": error_idx})
        error_data_array.encoding["zlib"] = True  # compress the data
        error_data_array.encoding["complevel"] = 6  # use compression level 6
        error_data_array.encoding["dtype"] = "S1"  # use compression level 6
        data_ararys["hex_errors"] = error_data_array

    return xr.Dataset(data_ararys)

def string_loader(path: Path, varname=None, encoding="CP437", content_md5=True) -> xr.Dataset:
    # This is not "read_text" to keep the same newline style as the input
    data_array = xr.DataArray(path.read_bytes().decode(encoding))
    data_array.attrs["filename"] = path.name
    if content_md5:
        data_array.attrs["Content-MD5"] = md5(path.read_bytes()).hexdigest()
    data_array.attrs["charset"] = encoding

    data_array.encoding["zlib"] = True  # compress the data
    data_array.encoding["complevel"] = 6  # use compression level 6
    data_array.encoding["dtype"] = "S1"
    return xr.Dataset({
        varname: data_array
    })

def read_hex(path, errors: ERRORS = "store", content_md5=True) -> xr.Dataset:
    path = Path(path)
    root = path.parent

    # this funny way of finding paths is so we don't need to care or guess about the case of the suffix/input
    # Patches welcome if there is a better way
    hex_path = list(root.glob(path.name, case_sensitive=False))

    xmlcon_name = Path(path.name).with_suffix(".xmlcon")
    bl_name = Path(path.name).with_suffix(".bl")
    hdr_name = Path(path.name).with_suffix(".hdr")

    xmlcon_path = list(root.glob(str(xmlcon_name), case_sensitive=False))
    bl_path = list(root.glob(str(bl_name), case_sensitive=False))
    hdr_path = list(root.glob(str(hdr_name), case_sensitive=False))

    # TODO: handle more then 1 found file for the above

    input_datasets = []
    if len(hex_path) == 1:
        input_datasets.append(hex_to_dataset(hex_path[0], errors=errors, content_md5=content_md5))

    if len(xmlcon_path) == 1:
        input_datasets.append(string_loader(xmlcon_path[0], "xmlcon", encoding="CP437", content_md5=content_md5))

    if len(bl_path) == 1:
        input_datasets.append(string_loader(bl_path[0], "bl", encoding="CP437", content_md5=content_md5))

    if len(hdr_path) == 1:
        input_datasets.append(string_loader(hdr_path[0], "hdr", encoding="CP437", content_md5=content_md5))

    return xr.merge(input_datasets)

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