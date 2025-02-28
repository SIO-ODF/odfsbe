"""
Module for parsing dataset data variables (.bl, .xmlcon, .hdr) into dictionaries
"""

import xml.etree.ElementTree as ET

import xarray as xr


def parse_bl(bl):
    """
    Parse the .bl file's text into lists of bottle fired and reset times.
    """
    bl_lines = bl.splitlines()
    resets = []
    log = []
    keys = ["sequence", "position", "time", "begining", "ending"]
    for line in bl_lines[1:]:
        if line.startswith("RESET"):
            resets.append(line.removeprefix("RESET "))
            continue
        cells = [cell.strip() for cell in line.split(",")]
        log.append(dict(zip(keys, cells, strict=True)))
    return log, resets


def parse_hdr(hdr):
    """
    Parse the .hdr file's text into a dictionary.
    """
    comments = []
    hdr_dict = {}
    for line in hdr.splitlines():
        if line.startswith("**"):
            comments.append(line.strip("* "))
            continue
        if line == "*END*":
            continue
        row = line.strip("* ")
        if row.startswith("Sea-Bird SBE 9"):
            continue
        if row.startswith("Software Version"):
            hdr_dict["Software Version"] = row.removeprefix("Software Version").strip()
            continue
        key, value = row.split("=")
        hdr_dict[key.strip()] = value.strip()
    hdr_dict["comments"] = "\n".join(comments)
    return hdr_dict


def parse_xmlcon(xml):
    """
    Parses an XMLCON configuration file into configuration and sensor metadata
    dictionaries.

    This function reads an XMLCON file, extracts deck unit configuration
    settings, and retrieves metadata and calibration coefficients for connected
    sensors.

    Code initially written 2024 Aaron Mau.

    Parameters
    ----------
    xml_config_path : str or pathlib.Path
        Path to the XMLCON file to be parsed.
    encoding : str, optional
        Encoding used to read the XMLCON file. Default is `"cp437"`.

    Returns
    -------
    tuple
        A tuple containing two dictionaries:
        - `config_dict` (dict): A dictionary of deck unit configuration
          settings, where keys are the configuration tags and values are their
          respective values.
        - `sensor_dict` (dict): A dictionary of sensor metadata, where keys are
          sensor indices (as integers) and values are nested dictionaries
          containing metadata and calibration coefficients for each sensor.

    Notes
    -----
    - The XMLCON file is parsed using an `ElementTree` object to extract both
      configuration settings and sensor information.
    - Calibration coefficients for sensors (e.g., `SBE4` or `SBE43`) are nested
      within the sensor metadata and are structured as dictionaries.
    - If a tag with calibration coefficients is repeated, it is appended with
      `2` to avoid overwriting.
    """
    if isinstance(xml, xr.DataArray):
        xml = xml.item() if xml.size == 1 else str(xml.values)

    #   Create xlmcon element tree
    xmlcon = ET.fromstring(xml)

    config_dict = {}  # Deck unit
    for deck_setting in xmlcon[0]:
        if not deck_setting.text.strip() or deck_setting.tag.strip() == "Name":
            # Preserve empty strings
            config_dict[deck_setting.tag.strip()] = deck_setting.text.strip()
        else:
            # Make other params integers/booleans
            config_dict[deck_setting.tag.strip()] = int(deck_setting.text.strip())
    sensor_array_size = xmlcon.find(
        ".//SensorArray"
    )  # Pull out reported sensor array size for comparison
    if sensor_array_size is not None:
        config_dict["SensorArraySize"] = int(sensor_array_size.attrib.get("Size"))

    sensor_dict = {}  # Sensors connected, with coefficients
    #   JB/JT endcap channel positions
    for position in xmlcon[0][-1]:
        #   Sensor metadata (SN, cal date, etc.)
        meta = {}
        meta["SensorID"] = position.attrib["SensorID"]
        for md_entry in position[0]:
            # print(md_entry.tag, md_entry.attrib, md_entry.text)
            #   Nested coefficients found in SBE4/SBE43
            if "Coefficients" in md_entry.tag:
                # print(
                #     f"Found nested calibration coefficients "
                #     f"in {md_entry.tag}")
                coefs = {}
                for coef in md_entry:
                    # print(coef.tag, coef.attrib, coef.text)
                    coefs[coef.tag.strip()] = coef.text.strip()
                if md_entry.tag in meta.keys():
                    meta[md_entry.tag.strip() + "2"] = coefs
                else:
                    meta[md_entry.tag.strip()] = coefs
            else:
                meta[md_entry.tag.strip()] = (
                    md_entry.text.strip() if md_entry.text else ""
                )

        sensor_dict[int(position.attrib["index"])] = meta

    if len(sensor_dict) != config_dict["SensorArraySize"]:
        print(
            f"Warning: XMLCON sensor dictionary size ({len(sensor_dict)})"
            f" differs from reported SensorArraySize "
            f"({config_dict['SensorArraySize']}). Check .XMLCON file."
        )

    return config_dict, sensor_dict
