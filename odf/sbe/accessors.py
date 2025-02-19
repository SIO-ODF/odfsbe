from pathlib import Path
from os import PathLike
from collections import ChainMap
from hashlib import md5

import xarray as xr
import numpy as np

from .channels import get_frequency, get_voltage

def write_path(data: bytes, path: Path, filename: str|None = None):
    if path.is_dir():
        with (path / filename).open("wb") as fo:
            fo.write(data)
    else:
        with path.open("wb") as fo:
            fo.write(data)


@xr.register_dataset_accessor("sbe")
class SBEAccessor():
    def __init__(self, xarray_object: xr.Dataset):
        self._obj = xarray_object

    def to_hex(self, path: str|PathLike|None = None, check=True):
        total_scans = self._obj.sizes["scan"]
        error_rows = {}
        if "hex_errors" in self._obj:
            total_scans += self._obj.sizes["scan_errors"]
            error_rows = dict(zip(self._obj.hex_errors.scan_errors.values.tolist(), self._obj.hex_errors.values.tolist(), ))

        header = "\r\n".join(self._obj.hex.attrs["header"].splitlines())
        data = bytes(self._obj.hex.as_numpy().values).hex().upper()
        row_len = self._obj.sizes["bytes_per_scan"] * 2
    
        data_rows = []
        scans = self._obj.scan.values
        for row in range(self._obj.sizes["scan"]):
            start = row * row_len
            stop = row * row_len + row_len
            data_rows.append((scans[row].item(), data[start:stop]))

        data_dict = ChainMap(dict(data_rows), error_rows)
        data_out = []
        for row in range(total_scans):
            scan = row + 1
            data_out.append(data_dict[scan])
        hex_data = "\r\n".join([header, *data_out, ""]).encode(self._obj.hex.attrs.get("charset", "utf8"))
        if check and (filehash := self._obj.hex.attrs.get("Content-MD5")) is not None:
            digest = md5(hex_data).hexdigest()
            if digest != filehash:
                raise ValueError("Output file does not match input")
        if path is None:
            return hex_data
        
        write_path(hex_data, Path(path), self._obj.hex.attrs["filename"])

    def to_hdr(self, path: str|PathLike|None = None, check=True):
        ...
    def to_xmlcon(self, path: str|PathLike|None = None, check=True):
        ...
    def to_bl(self, path: str|PathLike|None = None, check=True):
        ...

    def __getattr__(self, name):
        if name.startswith("f"):
            channel = int(name[1:])
            return get_frequency(self._obj.hex, channel)
        elif name.startswith("v"):
            channel = int(name[1:])
            return get_voltage(self._obj.hex, channel, 0)
        
        return super().__getattribute__(name)
