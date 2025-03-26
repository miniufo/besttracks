# -*- coding: utf-8 -*-
"""
Created on 2020.08.01

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
from typing import Union, Optional, Callable, List, Sequence, Any, Dict, Tuple
import pandas as pd
import xarray as xr
import numpy as np
import os
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from .core import TCSet, TC, Drifter, DrifterSet

undef = -9999.0


"""
Best-Track datasets are provided by several Regional Specialized Meteorological
Centers (RSMCs) or projects.  Frequently used datasets are those from:
1. Joint Typhoon Warning Center (JTWC), Naval Oceanography Portal
   URL: https://www.metoc.navy.mil/jtwc/jtwc.html?western-pacific
2. China Meteorological Administration (CMA),
   URL: http://tcdata.typhoon.org.cn/en/zjljsjj_sm.html
3. RSMC Tokyo-Typhoon Center, Japan Meteorological Agency (JMA)
   URL: https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/Besttracks/e_format_bst.html
4. National Hurricane Center (NHC), National Oceanic and Atmospheric
   Administration (NOAA):
   URL: https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-nencpac.pdf
5. International Best Track Archive for Climate Stewardship (IBTrACS)
   URL: https://www.ncdc.noaa.gov/ibtracs/pdf/IBTrACS_v04_column_documentation.pdf

In addition, we also provide the parser function for CMA operational forecast
data (BABJ format).

IO realted methods are defined below.
"""


def parse_TCs(
    filename: Union[str, Sequence[str]],
    rec_cond: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    tc_cond: Optional[Callable[[TC], bool]] = None,
    agency: str = 'CMA',
    wndunit: str = 'knot'
) -> TCSet:
    """
    Parse TC Best-Track data from various agents.

    Parameters
    ----------
    filename: str or sequence
        Either a string glob in the form ``"path/to/my/files/*.txt"`` or an
        explicit list of files to open.
    rec_cond: lambda expression
        Used by pandas.dataframe.loc[cond].  Example like:
        `lambda df:df['TIME'].dt.hour.isin([0, 6, 12, 18])`
    tc_cond: lambda expression
        Used to filter the TCs.  Example like:
        `lambda tc: tc.year == 2018`
    agency: str
        string for the data agencies, availables are
        ['JTWC', 'JMA', 'CMA', 'NHC', 'IBTrACS', 'BABJ'].
    wndunit: str
        Convert wind unit to wndunit, 'knot' or 'm/s'.

    Returns
    -------
    list of TC (data struct of a namedtuple)
    """
    agency = agency.upper()

    # Validate wind unit
    if wndunit not in ['knot', 'm/s']:
        raise ValueError(
            f'Invalid wind unit {wndunit}, should be "knot" or "m/s"')

    # Define parser mapping
    parser_mapping = {
        'CMA': parseCMA,
        'JMA': parseJMA,
        'JTWC': parseJTWC,
        'NHC': parseNHC,
        'IBTRACS': parseIBTrACS,
        'BABJ': parseBABJ
    }

    # Validate agency
    if agency not in parser_mapping:
        raise ValueError(
            f'Unsupported agency {agency}, should be one of {list(parser_mapping.keys())}')

    # Parse data using appropriate function
    data = parser_mapping[agency](filename)

    # Handle wind unit conversion
    conversion_factor = 0.51444  # Conversion factor: knot = m/s / 0.51444

    # Define agency groups by original wind unit
    knot_agencies = ['JMA', 'JTWC', 'NHC', 'IBTRACS']  # Provide wind in knots
    ms_agencies = ['CMA', 'BABJ']  # Provide wind in m/s

    # Perform wind unit conversion if needed
    if agency in ms_agencies and wndunit == 'knot':
        # Convert from m/s to knot
        mask = data['WND'] != undef
        data.loc[mask, 'WND'] = data.loc[mask, 'WND'] / conversion_factor
    elif agency in knot_agencies and wndunit == 'm/s':
        # Convert from knot to m/s
        mask = data['WND'] != undef
        data.loc[mask, 'WND'] = data.loc[mask, 'WND'] * conversion_factor

    # remove duplicated records
    data.drop_duplicates(inplace=True)

    # change dtypes to save memory
    data = data.astype({'LAT': np.float32, 'LON': np.float32,
                        'WND': np.float32, 'PRS': np.float32}, copy=False)

    if 'IDtmp' in data:
        # unique for CMA, JTWC
        TCdata = data.groupby('IDtmp')
    else:
        # unique for NHC, JMA, IBTrACS
        TCdata = data.groupby('ID')

    TCs: List[TC] = []

    for groupID, DATA in TCdata:
        DATA.reset_index(drop=True, inplace=True)

        # data cleaning using condition
        if rec_cond is not None:
            DATA = DATA.loc[rec_cond]
            DATA.reset_index(drop=True, inplace=True)

            if len(DATA) == 0:
                continue

        year: int = DATA.iloc[0]['TIME'].year
        name: str = DATA['NAME'].iloc[0]
        ID: str = DATA['ID'].iloc[0]

        if agency == 'BABJ':  # operational
            for time, fcstdata in DATA.groupby('TIME'):
                tc = TC(ID, name, year, wndunit, time,
                        fcstdata.drop(columns=['ID', 'NAME'])
                        .reset_index(drop=True))

                if tc_cond is None or tc_cond(tc):
                    TCs.append(tc)

        else:  # Best Track
            drops: List[str] = ['ID', 'NAME']

            if 'IDtmp' in DATA:
                drops.append('IDtmp')

            tc = TC(ID, name, year, wndunit, 0,
                    DATA.drop(columns=drops).reset_index(drop=True))

            if tc_cond is None or tc_cond(tc):
                TCs.append(tc)

    return TCSet(TCs, agency=agency)


def parse_GDPDrifters(filenames, full=False, chunksize=None, rawframe=False,
                      rec_cond=None, dr_cond=None):
    """
    Parse the drifter data file (ASCII) from NOAA into a pandas.DataFrame.

    Reference: https://www.aoml.noaa.gov/phod/gdp/buoydata_header.php

    Parameters
    ----------
    filename: str
        The file name of the CMA Best-Track data.
    full: bool
        Return full variables including uncertainties (default False).
    chunksize: int
        Chunk size for reading (default no chunk and read in all).
    rawframe: bool
        Return raw pandas.DataFrame or return drifters as a DrifterSet.
    rec_cond: lambda expression
        Used by pandas.dataframe.loc[cond].  Example like:
        `lambda df:df['TIME'].dt.hour.isin([0, 6, 12, 18])`
    dr_cond: lambda expression
        Used to filter the drifters.  Example like:
        `lambda dr: dr[0].LON < 120`

    Returns
    -------
    re: pandas.DataFrame
        Raw data in a pandas.DataFrame.
    """
    if full:
        cols = [(0, 8), (10, 25), (28, 35), (38,  45), (47,  56), (57, 65),
                (67, 75), (76, 85), (86, 98), (99, 111), (112, 124)]
        colN = ['ID', 'TIME', 'LAT', 'LON', 'SST', 'U', 'V', 'SPD',
                'VarLat', 'VarLon', 'VarT']
        dtyp = {'LAT': np.float32, 'LON': np.float32, 'Varlat': np.float32,
                'SST': np.float32, 'U': np.float32, 'Varlon': np.float32,
                'V': np.float32, 'SPD': np.float32, 'VarT': np.float32}
    else:
        cols = [(0,  8), (10, 25), (28, 35), (38, 45), (47, 56),
                (57, 65), (67, 75), (76, 85)]
        colN = ['ID', 'TIME', 'LAT', 'LON', 'SST', 'U', 'V', 'SPD']
        dtyp = {'LAT': np.float32, 'LON': np.float32, 'SST': np.float32,
                'U': np.float32, 'V': np.float32, 'SPD': np.float32}

    if chunksize == None:
        iters = [pd.read_fwf(f, header=None, colspecs=cols, names=colN,
                             parse_dates=['TIME'], dtype=dtyp,
                             date_parser=__parse_datetime)
                 for f in glob(filenames)]
    else:
        iters = [pd.read_fwf(f, header=None, colspecs=cols, names=colN,
                             parse_dates=['TIME'], iterator=True,
                             chunksize=chunksize, dtype=dtyp,
                             date_parser=__parse_datetime)
                 for f in glob(filenames)]

    data = pd.concat([chunk[rec_cond] for chunks in iters for chunk in chunks])

    if rawframe:
        return data

    data = data.astype(dtyp, copy=False)

    groups = data.groupby('ID')

    drifters = []

    for ID, records in groups:
        if len(records) < 2:
            raise Exception('only one record is found')
        elif len(records) == 2:
            if records['TIME'].diff().reset_index()['TIME'].loc[1] != \
                    pd.Timedelta(value=6, unit='H'):
                raise Exception('not 6hr interval\n' + str(records))
        else:
            freq = pd.infer_freq(records['TIME'])

        if freq != '6H':
            raise Exception('drifter (ID: {0:8d}) is not 6hr interval'
                            .format(freq))

        records.drop(columns='ID', inplace=True)
        # records = records.reset_index()

        dr = Drifter(str(ID), records)

        if dr_cond(dr):
            drifters.append(dr)

    return DrifterSet(drifters)


def parseNHC(filename, encoding='utf-8'):
    """
    Parse the Best-Track data from National Hurricane Center (NHC)
    into a pandas.DataFrame.

    Note that there are intensive observations to some recent TCs which are
    valid at 0300, 0900, 1500, 2100 UTCs.  These records can be filtered later.

    IDs for NHC is 4-digital year and 2-digital number within the year, since
    the records start on the year of 1851.

    Reference: https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-nencpac.pdf

    Parameters
    ----------
    filename: str
        The file name of the CMA Best-Track data.
    encoding: str
        Encoding of the file.

    Returns
    -------
    re: pandas.DataFrame
        Raw data in a pandas.DataFrame.
    """
    re = []

    with open(filename, 'r', encoding=encoding) as f:
        fileContent = f.readlines()

        for i, line in enumerate(fileContent):
            if len(line) < 50:
                tokens = line.split(',')

                count = tokens[2].strip()
                ID = tokens[0].strip()  # unique in NHC
                NAME = tokens[1].strip()

                ID = ID[4:8] + ID[2:4]

                if NAME == 'UNNAMED':
                    NAME = 'NONAME'

                count = int(count)

                strIdx = i + 1

                for ln in fileContent[strIdx:strIdx+count]:
                    TIME = datetime.strptime(ln[:8] + ln[10:12], "%Y%m%d%H")
                    TYPE = __get_type_NHC(ln[19:21])
                    LAT = float(ln[22:27].strip())
                    LON = float(ln[30:35].strip())
                    PRS = float(ln[43:47].strip())
                    WND = float(ln[38:41].strip())

                    if ln[35:36] == 'W':
                        LON = 360 - LON

                    if PRS == -999:
                        PRS = undef

                    if WND == -99 or WND == -999:
                        WND = undef

                    re.append((ID, NAME, TIME, LAT, LON, TYPE, PRS, WND))

    return pd.DataFrame.from_records(re, columns=['ID', 'NAME', 'TIME', 'LAT',
                                                  'LON', 'TYPE', 'PRS', 'WND'])


def parseIBTrACS(filename: Union[str, Path], encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Parse the Best-Track data from International Best Track Archive for Climate
    Stewardship (IBTrACS) into a pandas.DataFrame.

    Notice that only WMO-sanctioned variables are loaded.  Data from different
    agencies are ignored as they can be accessed from other databases.

    Reference: https://www.ncdc.noaa.gov/ibtracs/pdf/IBTrACS_v04_column_documentation.pdf

    Parameters
    ----------
    filename: str
        The file name of the IBTrACS Best-Track data.
    encoding: str
        Encoding of the file.

    Returns
    -------
    re: pandas.DataFrame
        Results in a pandas.DataFrame.
    """
    keeps: List[str] = ['sid', 'name', 'iso_time',
                        # 'numobs', 'basin', 'season',
                        'nature', 'lat', 'lon', 'wmo_wind', 'wmo_pres']

    KEEPS = [s.upper() for s in keeps]

    if filename.endswith('.nc'):
        ds = xr.open_dataset(filename)

        drops = [v for v in ds.variables]

        for keep in keeps:
            drops.remove(keep)

        ds = ds.drop_vars(drops)

        df = ds.to_dataframe().loc[lambda df: df['iso_time'] != b'']

        # rename columns
        renameDict = {'sid': 'ID',
                      # 'season': 'YEAR',
                      'name': 'NAME',
                      'iso_time': 'TIME',
                      'nature': 'TYPE',
                      'lat': 'LAT',
                      'lon': 'LON',
                      'wmo_wind': 'WND',
                      'wmo_pres': 'PRS'}

        df.reset_index(drop=True, inplace=True)

        df.loc[:, 'iso_time'] = pd.to_datetime(
            df['iso_time'].str.decode(encoding))
        df.loc[:, 'sid'] = df['sid'].str.decode(encoding)
        df.loc[:, 'name'] = df['name'].str.decode(encoding)
        df.loc[:, 'nature'] = df['nature'].str.decode(encoding)
        df.replace(np.nan, undef, inplace=True)

    elif filename.endswith('csv'):
        # rename columns
        renameDict = {'SID': 'ID',
                      # 'SEASON': 'YEAR',
                      'ISO_TIME': 'TIME',
                      'NATURE': 'TYPE',
                      'WMO_WIND': 'WND',
                      'WMO_PRES': 'PRS'}

        df = pd.read_csv(filename, usecols=KEEPS, skiprows=[1],
                         parse_dates=['ISO_TIME']).loc[lambda df: df['ISO_TIME'] != '']

        df.replace(r'^\s+$', undef, regex=True, inplace=True)

    else:
        raise ValueError(
            f"Unsupported file type: {filename}. Expected .nc or .csv")
    # Rename columns to standard format
    df.rename(columns=renameDict, inplace=True)

    return df


def parseCMA(filenames, encoding='utf-8'):
    """
    Parse the Best-Track data from China Meteorological Administration (CMA)
    into a pandas.DataFrame.

    Note that there are sub-center records which refers to a center that was
    split or induced from the original TC circulation center.  Currently, these
    records are removed during parsing.  Also there are some TCs without any
    official IDs (e.g., 0000) but have names.  These are modified to include
    the year as ID=199100.

    The maximum surface wind speed is defined as *2-min* averaged 10m wind
    speed in unit of *m/s*, in accordance with CMA standard.

    Reference: http://tcdata.typhoon.org.cn/en/zjljsjj_sm.html

    Parameters
    ----------
    filenames: str
        The file name(s) of the CMA Best-Track data.
    encoding: str
        Encoding of the file.

    Returns
    -------
    re: pandas.DataFrame
        Raw data (excluding sub-center records) in a pandas.DataFrame.
    """
    if isinstance(filenames, str):
        paths = sorted(glob(filenames))
    else:
        paths = [str(p) if isinstance(p, Path) else p for p in filenames]

    if not paths:
        raise OSError("no files to open")

    re = []

    for p in paths:
        fileContent = __concat_files(p, encoding=encoding)

        year = os.path.splitext(os.path.basename(p))[0][2:6]

        for i, line in enumerate(fileContent):
            if line.startswith('66666'):
                parts = line.split()

                count = parts[2].strip()
                IDtmp = parts[3].strip()  # this is unique in each year
                ID = parts[4].strip()  # official ID
                NAME = parts[7].strip()
                count = int(count)
                strIdx = i + 1

                # skip sub-center records.  The sub-center refers to a center
                # that was split or induced from the original tropical cyclone
                # circulation center.
                if NAME.find('(-)') != -1:
                    continue

                if NAME == '' or NAME == '(nameless)':
                    NAME = 'NONAME'

                # Parts of the records do not have official IDs (ID == '0000')
                # We modify them to contain at least information of year
                if ID == '0000':
                    ID = str(year) + '00'
                else:
                    # add century to ID
                    ID = str(year)[:2] + ID

                IDtmp = str(year) + IDtmp[2:]

                for ln in fileContent[strIdx:strIdx+count]:
                    tokens = ln.split()

                    TIME = datetime.strptime(tokens[0], "%Y%m%d%H")
                    TYPE = __get_type_CMA(tokens[1])
                    LAT = float(tokens[2]) / 10.0
                    LON = float(tokens[3]) / 10.0
                    PRS = float(tokens[4])
                    WND = float(tokens[5])  # m/s

                    re.append((IDtmp, ID, NAME, TIME,
                              LAT, LON, TYPE, PRS, WND))

    return pd.DataFrame.from_records(re, columns=['IDtmp', 'ID', 'NAME',
                                                  'TIME', 'LAT', 'LON', 'TYPE',
                                                  'PRS', 'WND'])


def parseJMA(filename: Union[str, Path], encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Parse the Best-Track data from Regional Specialized Meteorological
    Center (RSMC) Tokyo-Typhoon Center, Japan Meteorological Agency (JMA)
    into a pandas.DataFrame.

    Note that there are intensive observations to some recent TCs which are
    valid at 0300, 0900, 1500, 2100 UTCs. These records can be filtered later.

    The maximum surface wind speed is defined as *10-min* averaged 10m wind
    speed in unit of *knot*.

    Reference:
    https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/Besttracks/e_format_bst.html

    Parameters
    ----------
    filename : str or Path
        The file name of the JMA Best-Track data.
    encoding : str, default 'utf-8'
        Encoding of the file.

    Returns
    -------
    pd.DataFrame
        Parsed data with columns:
        ['ID', 'NAME', 'TIME', 'LAT', 'LON', 'TYPE', 'PRS', 'WND']

    Raises
    ------
    ValueError
        If the file format is invalid or IDs don't match.
    IOError
        If the file cannot be opened or read.
    """
    # Convert Path to string if needed
    if isinstance(filename, Path):
        filename = str(filename)
        
    records = []

    try:
        with open(filename, 'r', encoding=encoding) as f:
            # Read all lines at once - more efficient than reading line by line
            file_content = f.readlines()

            for i, line in enumerate(file_content):
                if line.startswith('66666'):
                    # Parse header information
                    count_str = line[12:15].strip()
                    ID = line[21:25].strip()  # unique ID
                    IDtmp = line[6:10].strip()
                    NAME = line[30:50].strip() or 'NONAME'  # Default to 'NONAME' if empty
                    
                    try:
                        count = int(count_str)
                    except ValueError:
                        print(f"Warning: Invalid count value '{count_str}' at line {i+1}, skipping")
                        continue
                    
                    # Start index of data records
                    strIdx = i + 1
                    
                    # Check if IDs match
                    if ID != IDtmp:
                        raise ValueError(f"IDs don't match: {ID} != {IDtmp} at line {i+1}")

                    # Only process if we have at least one data record
                    if strIdx < len(file_content):
                        # Get the year from the first record
                        tokens = file_content[strIdx].split()
                        if not tokens:
                            continue
                            
                        try:
                            time_obj = datetime.strptime(tokens[0], "%y%m%d%H")
                            year = time_obj.year
                            
                            # Handle years in the range 1951-1968
                            if 51 <= int(tokens[0][:2]) <= 68:
                                year -= 100
                                
                            # Add century to ID
                            century_prefix = str(year)[:2]
                            ID_with_century = century_prefix + ID
                            
                            # Process all data records for this TC
                            for j in range(strIdx, min(strIdx + count, len(file_content))):
                                tokens = file_content[j].split()
                                if len(tokens) < 6:
                                    continue
                                
                                # Parse data fields
                                TIME = datetime.strptime(tokens[0], "%y%m%d%H")
                                
                                # Handle years in the range 1951-1968
                                if 51 <= int(tokens[0][:2]) <= 68:
                                    TIME = TIME.replace(year=TIME.year-100)
                                    
                                TYPE = __get_type_JMA(tokens[2])
                                LAT = float(tokens[3]) / 10.0
                                LON = float(tokens[4]) / 10.0
                                PRS = float(tokens[5])
                                
                                # Wind data may be missing
                                WND = float(tokens[6]) if len(tokens) > 6 else undef
                                
                                records.append((ID_with_century, NAME, TIME, LAT, LON, TYPE, PRS, WND))
                        except (ValueError, IndexError) as e:
                            print(f"Error parsing data at line {strIdx+1}: {str(e)}")
    except IOError as e:
        raise IOError(f"Could not read file {filename}: {str(e)}")

    columns = ['ID', 'NAME', 'TIME', 'LAT', 'LON', 'TYPE', 'PRS', 'WND']
    result_df = pd.DataFrame.from_records(records, columns=columns)
    
    return result_df


def parseJTWC(
    filenames: Union[str, Sequence[Union[str, Path]]],
    encoding: str = 'utf-8'
) -> pd.DataFrame:
    """
    Parse the Best-Track data from Joint Typhoon Warning Center (JTWC)
    into a pandas.DataFrame.

    Note that there is no official ID and name for each TC. So the ID is
    a simple composite of year + number of TC, while the name is the
    composite of basin (e.g., WP) + ID. Also there are some duplicated data.

    The maximum surface wind speed is defined as *1-min* averaged 10m wind
    speed in unit of *knot*.

    Possible errors in the original txt files:
    - bwp231969.txt, time 1969100000 should be 1969100100?

    Parameters
    ----------
    filenames : str or sequence of str or Path
        The file name(s) or pattern of the JTWC Best-Track data.
    encoding : str, default 'utf-8'
        Encoding of the files.

    Returns
    -------
    pd.DataFrame
        Parsed data with columns:
        ['IDtmp', 'ID', 'NAME', 'TIME', 'LAT', 'LON', 'PRS', 'WND']

    Raises
    ------
    OSError
        If no files match the provided pattern.
    """
    # Handle input file patterns/paths
    if isinstance(filenames, str):
        paths = sorted(glob(filenames))
    else:
        paths = [str(p) if isinstance(p, Path) else p for p in filenames]

    if not paths:
        raise OSError(f"No files found matching pattern: {filenames}")

    # Pre-allocate list for records with estimated capacity
    records = []
    
    for path in paths:
        try:
            # Read file content
            file_content = __concat_files(path, encoding=encoding)
            
            if not file_content:
                print(f'Empty file found: {path}')
                continue
                
            # Extract ID from filename
            file_basename = os.path.splitext(os.path.basename(path))[0]
            
            # Parse year prefix and TC number
            year_prefix = file_content[0][10:12]
            tc_number = file_content[0][4:6]
            
            # Determine full ID with century
            if int(year_prefix) >= 45:  # Pre-2000s
                full_id = f"19{year_prefix}{tc_number}"
            else:
                full_id = f"20{year_prefix}{tc_number}"
            
            # Process all lines in the file
            for line in file_content:
                # Parse data fields with specific positions
                time_str = line[8:18]
                lat_str = line[35:38]
                lat_hem = line[38:39]
                lon_str = line[41:45]
                lon_hem = line[45:46]
                wind_str = line[47:51]
                pres_str = line[52:57].strip()
                
                try:
                    # Convert fields to appropriate types
                    time = datetime.strptime(time_str, "%Y%m%d%H")
                    lat = float(lat_str) / 10.0
                    lon = float(lon_str) / 10.0
                    wind = float(wind_str)
                    
                    if lat_hem == 'S':
                        lat = -lat
                    if lon_hem == 'W':
                        lon = 360.0 - lon
                    
                    pres = float(pres_str) if pres_str else undef
                    
                    if wind == -999:
                        wind = undef
                    
                    records.append((file_basename, full_id, 'NONAME', time, lat, lon, pres, wind))
                    
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line in {path}: {line.strip()}\n{str(e)}")
                
        except Exception as e:
            print(f"Error processing file {path}: {str(e)}")
    
    columns = ['IDtmp', 'ID', 'NAME', 'TIME', 'LAT', 'LON', 'PRS', 'WND']
    result_df = pd.DataFrame.from_records(records, columns=columns)
    
    return result_df


def parseBABJ(
    filenames: Union[str, Sequence[Union[str, Path]]],
    encoding: str = 'GBK'
) -> pd.DataFrame:
    """
    Parse China Meteorological Administration (CMA) operational forecast data
    (babj format) into a pandas.DataFrame.

    Note that the data use Beijing time and they are changed to UTC during the
    parsing. Records with undefined pressure data (PRS=9999.0) are removed.

    Parameters
    ----------
    filenames: str or sequence of str or Path
        The file name pattern or list of files containing BABJ format data.
    encoding: str, default 'GBK'
        Character encoding of the input files.

    Returns
    -------
    pd.DataFrame
        Parsed forecast data with columns:
        ['ID', 'NAME', 'TIME', 'FCST', 'LAT', 'LON', 'PRS', 'WND']
        where TIME is converted from Beijing time to UTC.

    Raises
    ------
    ValueError
        If no files match the provided pattern or if file format is invalid.
    """
    paths = sorted(glob(filenames))

    if not paths:
        raise ValueError(f"No files found matching pattern: {filenames}")

    records = []

    for path in paths:
        try:
            with open(path, 'r', encoding=encoding) as f:
                file_content = f.readlines()

                if len(file_content) < 2:
                    print(
                        f"Warning: File {path} has insufficient data, skipping")
                    continue

                tokens = file_content[1].split()
                if len(tokens) < 4:
                    print(f"Warning: Invalid header in {path}, skipping")
                    continue

                name = tokens[0]
                tc_id = tokens[1]
                try:
                    count = int(tokens[3])
                except ValueError:
                    print(f"Warning: Invalid count value in {path}, skipping")
                    continue

                start_idx = 3
                end_idx = min(start_idx + count, len(file_content))

                for line in file_content[start_idx:end_idx]:
                    tokens = line.split()
                    if len(tokens) < 9:
                        continue

                    time_str = tokens[0] + tokens[1] + tokens[2] + tokens[3]

                    try:
                        time = datetime.strptime(
                            time_str, "%Y%m%d%H") - timedelta(hours=8)
                        fcst = int(tokens[4])
                        lon = float(tokens[5])
                        lat = float(tokens[6])
                        prs = float(tokens[7])
                        wnd = float(tokens[8])

                        if prs != 9999.0:
                            records.append(
                                (tc_id, name, time, fcst, lat, lon, prs, wnd))
                        else:
                            print(
                                f"Skipping record with undefined pressure in {tc_id}: {line.strip()}")
                    except (ValueError, IndexError) as e:
                        print(
                            f"Error parsing line in {path}: {line.strip()}\n{str(e)}")

        except Exception as e:
            print(f"Error processing file {path}: {str(e)}")

    columns = ['ID', 'NAME', 'TIME', 'FCST', 'LAT', 'LON', 'PRS', 'WND']
    result_df = pd.DataFrame.from_records(records, columns=columns)

    return result_df


"""
Helper (private) methods are defined below
"""


def __concat_files(paths: Union[str, Sequence[Union[str, Path]]], encoding: str = 'utf-8') -> List[str]:
    """

    Parameters
    ----------
    paths : str or sequence
        Either a string glob in the form ``"path/to/my/files/*.ctl"`` or an
        explicit list of files to open. Paths can be given as strings or as
        pathlib Paths.
    encoding : str
        Encoding for the ctl file content e.g., ['GBK', 'UTF-8'].

    Returns
    -------
    re: str
        A single string containing all the content of files.
    """
    if isinstance(paths, str):
        paths = sorted(glob(paths))
    else:
        paths = [str(p) if isinstance(p, Path) else p for p in paths]

    if not paths:
        raise OSError("No files to open")

    result = []
    for filepath in paths:
        with open(filepath, 'r', encoding=encoding) as f:
            lines = [line if line.endswith(
                '\n') else line + '\n' for line in f]
            result.extend(lines)

    return re


def __get_type_JMA(code: str) -> str:
    """
    Get the type (grade) of the JMA TC records.
    Reference:
    https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/Besttracks/e_format_bst.html

    1 - not used
    2 - tropical depression (TD)
    3 - tropical storm (TS)
    4 - severe tropical storm (STS)
    5 - typhoon (TY)
    6 - extra-tropical cyclone (EC)
    7 - just entering into the responsible area of JMA
    8 - not used
    9 - tropical cyclone of TC intensity or higher

    Parameters
    ----------
    code : str
        A string code represents the type.

    Returns
    -------
    re : str
        One of the type in ['TD', 'TS', 'STS', 'TY', 'EC', 'OTHERS'].
    """
    type_mapping = {
        '1': 'OTHERS',
        '2': 'TD',
        '3': 'TS',
        '4': 'STS',
        '5': 'TY',
        '6': 'EC',
        '7': 'OTHERS',
        '8': 'OTHERS',
        '9': 'OTHERS'
    }

    try:
        return type_mapping[code]
    except KeyError:
        raise ValueError(f'Unknown code: {code}')


def __get_type_CMA(code: str) -> str:
    """
    Get the intensity category according to "Chinese National Standard for
    Grade of Tropical Cyclones", which was put in practice since 15 June 2006.

    Reference:
    http://tcdata.typhoon.org.cn/zjljsjj_sm.html

    0 - weaker than TD or unknown
    1 - tropical depression	  (TD , 10.8-17.1 m/s)
    2 - tropical storm	      (TS , 17.2-24.4 m/s)
    3 - severe tropical storm (STS, 24.5-32.6 m/s)
    4 - typhoon               (TY , 32.7-41.4 m/s)
    5 - severe typhoon        (STY, 41.5-50.9 m/s)
    6 - super typhoon         (superTY, >=51.0 m/s)
    9 - extratropical cyclone (EC)

    Parameters
    ----------
    code : str
        A string code represents the type.

    Returns
    -------
    re : str
        One of the type in ['TD', 'TS', 'STS', 'TY', 'STY', 'EC', 'OTHERS'].
    """
    type_mapping = {
        '0': 'OTHERS',
        '1': 'TD',
        '2': 'TS',
        '3': 'STS',
        '4': 'TY',
        '5': 'STY',
        '6': 'STY',
        '9': 'EC'
    }

    try:
        return type_mapping[code]
    except KeyError:
        raise ValueError(f'Unknown code: {code}')


def __get_type_NHC(code):
    """
    Get the intensity category according to the status of system defined by
    "National Hurricane Center".

    Reference:
    https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-nov2019.pdf

    0 - Subtropical cyclone of depression intensity;
        Subtropical cyclone of storm intensity;
        A low that is neither a TC, a subtropical cyclone, nor an EC;
        Tropical wave;
        Disturbuance          (OTHERS, unknown intensity)
        1 - Tropical depression	  (TD,   <34 knots)
        2 - Tropical storm	      (TS, 34-63 knots)
        3 - Hurricane             (HU,   >64 knots)
        4 - Extratropical cyclone (EC, any intensity)

    Parameters
    ----------
    code : str
        A string code represents the type.

    Returns
    -------
    re: str
        One of the type in ['TD', 'TS', 'HU', 'EC', 'OTHERS'].
    """
    return code


def __parse_datetime(text: str) -> pd.Timestamp:
    """
    Parse datetime string from GDP drifter data format into pandas Timestamp.

    Handles specific format: "MM DD.ddd YYYY" where ddd represents fractional day:
    - .000 = 00:00 (0 hours)
    - .250 = 06:00 (6 hours)
    - .500 = 12:00 (12 hours)
    - .750 = 18:00 (18 hours)

    Reference: https://www.aoml.noaa.gov/phod/gdp/buoydata_header.php

    Parameters
    ----------
    text: str
        String text for datetime in GDP drifter format (e.g., "1 15.000 1979")

    Returns
    -------
    pd.Timestamp
        A pandas Timestamp object representing the parsed datetime

    Raises
    ------
    ValueError
        If the hour fraction is invalid (not in [.000, .250, .500, .750])
    """
    hour_mapping = {
        '000': 0,
        '250': 6,
        '500': 12,
        '750': 18
    }

    parts = text.split()
    month = int(parts[0])

    day_parts = parts[1].split('.')
    day = int(day_parts[0])
    hour_code = day_parts[1]
    year = int(parts[2])

    try:
        hour = hour_mapping[hour_code]
    except KeyError:
        raise ValueError(
            f'Invalid hour code: {hour_code}, must be one of {list(hour_mapping.keys())}')

    return pd.Timestamp(year=year, month=month, day=day, hour=hour)


"""
Test codes
"""
if __name__ == '__main__':
    print('ok')

    re = __concat_files('D:/Data/Typhoons/CMA/original/CH1949BST.txt')

    print(re)
