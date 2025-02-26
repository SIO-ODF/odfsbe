# odf.sbe
[![Tests CI Status](https://github.com/SIO-ODF/odfsbe/actions/workflows/run-tests.yml/badge.svg)](https://github.com/SIO-ODF/odfsbe/actions/workflows/run-tests.yml)
[![Doc Build Status](https://github.com/SIO-ODF/odfsbe/actions/workflows/build-docs.yml/badge.svg)](https://sio-odf.github.io/odfsbe/)
[![Coverage Status](https://coveralls.io/repos/github/SIO-ODF/odfsbe/badge.svg?branch=master)](https://coveralls.io/github/SIO-ODF/odfsbe?branch=master)

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

> [!NOTE]
> The name isn't final, it was quickly made during the GP17-ANT cruise
> but not developed further on that trip. Since this will likely be the
> base of some ODF/CCDHO/R2R CTD processing, I wanted it off my laptop
> as the sole location. -Barna
