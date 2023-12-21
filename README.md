# odf.sbe
A raw seabird hex + xml + bl to xarray/netCDF 4 and back again library

## Goals
* hex + xmlcon + bl file to xarray (and by extension, netCDF4)
* xarray to hex + xmlcon + bl (byte for byte reproduction is something we should test for)
- include some way to take out bad scans/data lines, but put them back in for round tripping
* xarray "hex" to engineering variables, we might want to do this with an accesssor in the library for now since the conversion is so fast...
* tests (pytest)
* typing (mypy)
* documentation (sphinx)
- some description of the netCDF data structure...

Stretch goals:
* Making a btl file
* interpreting the modulo and the sample frequency to make actual timestamped time series variables