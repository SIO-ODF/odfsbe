from pathlib import Path
from os import PathLike
from hashlib import md5
from collections import ChainMap

import xarray as xr

from .channels import get_frequency, get_voltage
from odf.sbe.io import write_path, string_writer


@xr.register_dataset_accessor("sbe")
class SBEAccessor:
    def __init__(self, xarray_object: xr.Dataset):
        self._obj = xarray_object

    def to_hex(self, path: str | PathLike | None = None, check=True):
        """return or write a .hex file identical to the input of read_hex

        The output will not be identical if the errors of read_hex was not "store".
        """
        # extract the two relivant DataArrays
        # Note that the scan coordinate will be carried with these by xarray
        _hex = self._obj.hex
        _hex_errors = self._obj.get("hex_errors")

        # If there are errors rows, construct a dict that maps scan count to hex string
        # Also update the total scan to be the sum of normal scans and error scans
        error_rows = {}
        total_scans = _hex.sizes["scan"]
        if _hex_errors is not None:
            total_scans += _hex_errors.sizes["scan_errors"]
            error_rows = dict(
                zip(
                    _hex_errors.scan_errors.values.tolist(),
                    _hex_errors.values.tolist(),
                )
            )

        # just prepare the header
        header = "\r\n".join(_hex.attrs["header"].splitlines())

        # construct a dict that maps scan count to hex string
        # the hex_data is made as one big string all at once (very fast)
        # so the index/pointer math quickly slices this into rows
        data_rows = []
        scans = _hex.scan.values
        row_len = _hex.sizes["bytes_per_scan"] * 2
        hex_data = bytes(_hex.as_numpy().values).hex().upper()
        for row in range(_hex.sizes["scan"]):
            start = row * row_len
            stop = row * row_len + row_len
            data_rows.append((scans[row].item(), hex_data[start:stop]))

        # Chain map will check the dicts in order for the presence of the key
        # data out is the "final" list of hex strings that will be joined by the line seperator
        data_dict = ChainMap(dict(data_rows), error_rows)
        data_out = []
        for row in range(total_scans):
            scan = row + 1
            data_out.append(data_dict[scan])
        # The final "" here makes an empty line at the end of the file
        data_out = "\r\n".join([header, *data_out, ""]).encode(
            _hex.attrs.get("charset", "utf8")
        )
        # do the output check but only if there is a Content-MD5 attr
        if check and (filehash := _hex.attrs.get("Content-MD5")) is not None:
            digest = md5(data_out).hexdigest()
            if digest != filehash:
                raise ValueError("Output file does not match input")
        if path is None:
            return data_out

        write_path(data_out, Path(path), _hex.attrs["filename"])

    def _str_to_bytes_or_file(
        self, var, path: str | PathLike | None = None, check=True
    ):
        _var = self._obj[var]

        data_out = string_writer(_var, check=check)

        if path is None:
            return data_out

        write_path(data_out, Path(path), _var.attrs["filename"])

    def to_hdr(self, path: str | PathLike | None = None, check=True):
        return self._str_to_bytes_or_file("hdr", path=path, check=check)

    def to_xmlcon(self, path: str | PathLike | None = None, check=True):
        return self._str_to_bytes_or_file("xmlcon", path=path, check=check)

    def to_bl(self, path: str | PathLike | None = None, check=True):
        return self._str_to_bytes_or_file("bl", path=path, check=check)

    def all_to_dir(self, path: str | PathLike, check=True):
        """Write all possible output files to path

        Given some path to a directory, will export all the files (hex, xmlcon, bl, hdr) using their input filenames.
        """
        _path = Path(path)
        if not _path.is_dir():
            raise ValueError(f"{path} must be a directory")
        if "hex" in self._obj:
            self.to_hex(_path, check=check)
        if "xmlcon" in self._obj:
            self.to_xmlcon(_path, check=check)
        if "hdr" in self._obj:
            self.to_hdr(_path, check=check)
        if "bl" in self._obj:
            self.to_bl(_path, check=check)

    def __getattr__(self, name):
        if name.startswith("f"):
            channel = int(name[1:])
            return get_frequency(self._obj.hex, channel)
        elif name.startswith("v"):
            channel = int(name[1:])
            return get_voltage(self._obj.hex, channel, 0)

        return super().__getattribute__(name)
