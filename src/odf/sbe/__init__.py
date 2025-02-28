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

# from odf.sbe.io import read_hex
from odf.sbe.io import read_hex

__all__ = ["read_hex"]
