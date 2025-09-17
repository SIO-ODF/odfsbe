from hashlib import md5
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr

ERRORS = Literal["store", "raise", "ignore"]


def write_path(data: bytes, path: Path, filename: str | None = None):
    if path.is_dir() and filename is not None:
        with (path / filename).open("wb") as fo:
            fo.write(data)
    else:
        with path.open("wb") as fo:
            fo.write(data)


def string_writer(da: xr.DataArray, check=True) -> bytes:
    _item = da.values.item()
    encoding = da.attrs.get("charset", "utf8")

    _out = _item.encode(encoding)

    if check and (filehash := da.attrs.get("Content-MD5")) is not None:
        digest = md5(_out).hexdigest()
        if digest != filehash:
            raise ValueError("Output file does not match input")

    return _out


def hex_to_dataset(
    path: Path, errors: ERRORS = "store", encoding="CP437", content_md5=True
) -> xr.Dataset:
    out = bytearray()  #   hex bytes out
    hex = path.read_text(encoding)

    a = np.array(hex.splitlines())
    comments_i = np.strings.startswith(a, "*")
    data_i = ~comments_i

    comments = a[comments_i]
    data = a[data_i]
    data_lens = np.strings.str_len(data)

    # scan length?
    scans = np.arange(data.shape[0]) + 1
    comment_scan_length = (
        np.strings.find(np.strings.lower(comments), "number of bytes per scan") > -1
    )
    # we want exactly one line that matches this
    if comment_scan_length.sum() == 1:
        line = comments[comment_scan_length]
        line_length = int(line.item().split("= ")[1]) * 2
    else:  # calcualte from the most common line length in data
        counts_ = np.unique_counts(data_lens)
        line_length = counts_.values[np.argmax(counts_.counts)]

    ok_data_i = data_lens == line_length
    ok_data = data[ok_data_i]
    ok_scans = scans[ok_data_i]

    bad_data_i = ~ok_data_i
    bad_data = data[bad_data_i]
    bad_scans = scans[bad_data_i]

    if len(bad_data) > 0 and errors == "raise":
        raise ValueError("invalid scan lengths")

    out = np.frombuffer(bytes.fromhex("".join(ok_data)), dtype=np.uint8).reshape(
        (ok_data.shape[0], line_length // 2)
    )

    data_array = xr.DataArray(
        out, dims=["scan", "bytes_per_scan"], coords={"scan": ok_scans}
    )
    data_array.attrs["header"] = (
        "\n".join(comments)
        # header  # utf8 needs to be encoded using .attrs["charset"] when written back out
    )

    data_array.attrs["filename"] = path.name
    if content_md5:
        data_array.attrs["Content-MD5"] = md5(path.read_bytes()).hexdigest()
    data_array.attrs["charset"] = encoding

    # Encoding is instructions for xarray
    data_array.encoding["zlib"] = True  # compress the data
    data_array.encoding["complevel"] = 6  # use compression level 6
    data_array.encoding["chunksizes"] = (
        60 * 60 * 24,
        1,
    )  # chunk every hour of data (for 24hz data), and each column seperately
    # This is about 3~4mb chunks uncompressed depending on how many channels there are
    data_ararys = {"hex": data_array}

    if errors == "store" and len(bad_data) > 0:
        # make a string array of the bad lines
        error_data_array = xr.DataArray(
            bad_data, dims=["scan_errors"], coords={"scan_errors": bad_scans}
        )
        error_data_array.encoding["zlib"] = True  # compress the data
        error_data_array.encoding["complevel"] = 6  # use compression level 6
        error_data_array.encoding["dtype"] = "S1"  # use compression level 6
        data_ararys["hex_errors"] = error_data_array

    return xr.Dataset(data_ararys)


def string_loader(
    path: Path, varname=None, encoding="CP437", content_md5=True
) -> xr.Dataset:
    # This is not "read_text" to keep the same newline style as the input
    data_array = xr.DataArray(path.read_bytes().decode(encoding))
    data_array.attrs["filename"] = path.name
    if content_md5:
        data_array.attrs["Content-MD5"] = md5(path.read_bytes()).hexdigest()
    data_array.attrs["charset"] = encoding

    data_array.encoding["zlib"] = True  # compress the data
    data_array.encoding["complevel"] = 6  # use compression level 6
    data_array.encoding["dtype"] = "S1"
    return xr.Dataset({varname: data_array})


def read_hex(path, errors: ERRORS = "store", content_md5=True) -> xr.Dataset:
    path = Path(path)
    root = path.parent

    # this funny way of finding paths is so we don't need to care or guess about the case of the suffix/input
    # Patches welcome if there is a better way
    hex_path = list(root.glob(path.name, case_sensitive=False))

    xmlcon_name = Path(path.name).with_suffix(".xmlcon")
    bl_name = Path(path.name).with_suffix(".bl")
    hdr_name = Path(path.name).with_suffix(".hdr")
    mrk_name = Path(path.name).with_suffix(".mrk")

    xmlcon_path = list(root.glob(str(xmlcon_name), case_sensitive=False))
    bl_path = list(root.glob(str(bl_name), case_sensitive=False))
    hdr_path = list(root.glob(str(hdr_name), case_sensitive=False))
    mrk_path = list(root.glob(str(mrk_name), case_sensitive=False))

    # TODO: handle more then 1 found file for the above

    input_datasets = []
    if len(hex_path) == 1:
        input_datasets.append(
            hex_to_dataset(hex_path[0], errors=errors, content_md5=content_md5)
        )

    if len(xmlcon_path) == 1:
        input_datasets.append(
            string_loader(
                xmlcon_path[0], "xmlcon", encoding="CP437", content_md5=content_md5
            )
        )

    if len(bl_path) == 1:
        input_datasets.append(
            string_loader(bl_path[0], "bl", encoding="CP437", content_md5=content_md5)
        )

    if len(hdr_path) == 1:
        input_datasets.append(
            string_loader(hdr_path[0], "hdr", encoding="CP437", content_md5=content_md5)
        )

    if len(mrk_path) == 1:
        input_datasets.append(
            string_loader(mrk_path[0], "mrk", encoding="CP437", content_md5=content_md5)
        )

    return xr.merge(input_datasets)
