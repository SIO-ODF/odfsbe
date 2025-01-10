import xarray as xr

from .channels import get_frequency, get_voltage

@xr.register_dataset_accessor("sbe")
class SBEAccessor():
    def __init__(self, xarray_object: xr.Dataset):
        self._obj = xarray_object

    def __getattr__(self, name):
        if name.startswith("f"):
            channel = int(name[1:])
            return get_frequency(self._obj.hex, channel)
        elif name.startswith("v"):
            channel = int(name[1:])
            return get_voltage(self._obj.hex, channel, 0)
        
        return super().__getattribute__(name)
