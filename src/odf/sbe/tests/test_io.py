import xarray as xr

import odf.sbe.io as odf_io


def test_read_hex(tmp_path):
    # Create test files
    hex_content = """* Test header\nnumber of bytes per scan = 2\n0102\n0304"""
    xmlcon_content = "<xml>Test XMLCON</xml>"
    bl_content = "Test BL content"  #   mrk-like
    hdr_content = "Test HDR content"
    mrk_content = "Test MRK content"  #   bl-like

    # Define file paths
    hex_path = tmp_path / "test_file.hex"
    xmlcon_path = tmp_path / "test_file.xmlcon"
    bl_path = tmp_path / "test_file.bl"
    hdr_path = tmp_path / "test_file.hdr"
    mrk_path = tmp_path / "test_file.mrk"

    # Write files to the tmp_path and run
    hex_path.write_text(hex_content)
    xmlcon_path.write_text(xmlcon_content)
    bl_path.write_text(bl_content)
    hdr_path.write_text(hdr_content)
    mrk_path.write_text(mrk_content)
    dataset = odf_io.read_hex(hex_path)

    # Check contents
    assert isinstance(dataset, xr.Dataset)
    assert "hex" in dataset
    assert "xmlcon" in dataset
    assert "bl" in dataset
    assert "hdr" in dataset
    assert "mrk" in dataset

    # Confirm it's what we put in
    assert dataset["xmlcon"].values.item() == xmlcon_content
    assert dataset["bl"].values.item() == bl_content
    assert dataset["hdr"].values.item() == hdr_content
    assert dataset["mrk"].values.item() == mrk_content

    # Test with a missing non-.hex file
    hdr_path.unlink()  # Remove the .hdr file
    dataset = odf_io.read_hex(hex_path)
    assert "hdr" not in dataset
