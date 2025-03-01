from collections import ChainMap
from collections.abc import Mapping
from functools import cached_property
from hashlib import md5
from os import PathLike
from pathlib import Path

import xarray as xr

from odf.sbe.io import string_writer, write_path

from .channels import (
    get_frequency as _get_frequency,
)
from .channels import (
    get_metadata,
)
from .channels import (
    get_voltage as _get_voltage,
)
from .parsers import parse_xmlcon


@xr.register_dataset_accessor("sbe")
class SBEAccessor(Mapping):
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
                    strict=True,
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
        list_out = []
        for row in range(total_scans):
            scan = row + 1
            list_out.append(data_dict[scan])
        # The final "" here makes an empty line at the end of the file
        data_out = "\r\n".join([header, *list_out, ""]).encode(
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

    def _xmlcon(self):
        return parse_xmlcon(self._obj.xmlcon)

    def get_frequency(self, freq: int):
        if freq + 1 > self.num_frequencies:
            raise IndexError()
        return _get_frequency(self._obj.hex, freq)

    def get_voltage(self, voltage: int):
        frequencies_suppressed = self.config["FrequencyChannelsSuppressed"]
        if voltage + 1 > self.num_voltages:
            raise IndexError()
        return _get_voltage(self._obj.hex, voltage, frequencies_suppressed)

    @property
    def num_frequencies(self) -> int:
        return 5 - self.config["FrequencyChannelsSuppressed"]

    @property
    def num_voltages(self) -> int:
        return 8 - self.config["VoltageWordsSuppressed"]

    @property
    def config(self):
        return self._xmlcon()[0]

    @property
    def sensors(self):
        return self._xmlcon()[1]

    def serialize(self):
        new_obj = self._obj.copy()
        new_obj.update(self)
        return new_obj

    @cached_property
    def _meta(self):
        """
        Write out metadata columns using the wrapper
        """
        return get_metadata(self._obj.hex, self.config)

    @property
    def _names(self):
        frequencies = [f"f{idx}" for idx in range(self.num_frequencies)]
        voltages = [f"v{idx}" for idx in range(self.num_voltages)]
        return [*frequencies, *voltages, *self._meta]

    def __getitem__(self, name):
        if name not in self._names:
            raise KeyError()
        if name in self._meta:
            return self._meta[name]
        elif name.startswith("f"):
            channel = int(name[1:])
            return self.get_frequency(channel)
        elif name.startswith("v"):
            channel = int(name[1:])
            return self.get_voltage(channel)

    def __len__(self):
        return len(self._names)

    def __iter__(self):
        return iter(self._names)
