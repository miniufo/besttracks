# -*- coding: utf-8 -*-
"""
Created on 2020.08.01

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import pandas as pd
import xarray as xr
import numpy as np
import os
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from .core import TCSet, TC

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
def parse_TCs(filename, cond=None, agency='CMA', wndunit='knot'):
    """
    Parse TC Best-Track data from various agents.

    Parameters
    ----------
    filename: str or sequence
        Either a string glob in the form ``"path/to/my/files/*.txt"`` or an
        explicit list of files to open.
    cond: (lambda) expression
        Used by pandas.dataframe.loc[cond].  Example like:
        `lambda df:df['TIME'].dt.hour.isin([0, 6, 12, 18])`
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
    
    if wndunit not in ['knot', 'm/s']:
        raise Exception('invalid wind unit {0:s}, should be "knot" or "m/s"'
                        .format(wndunit))

    if agency == 'CMA': ############## CMA ###############
        data = parseCMA(filename)
        # change wind unit to knot
        if wndunit == 'knot':
            data['WND'].where(data['WND']==undef,
                              data['WND']/0.51444,
                              inplace=True)
        
    elif agency == 'JMA': ############ JMA ###############
        data = parseJMA(filename)
        # change wind unit to m/s
        if wndunit == 'm/s':
            data['WND'].where(data['WND']==undef,
                              data['WND']*0.51444,
                              inplace=True)
            
    elif agency == 'JTWC': ########### JTWC ##############
        data = parseJTWC(filename)
        # change wind unit to m/s
        if wndunit == 'm/s':
            data['WND'].where(data['WND']==undef,
                              data['WND']*0.51444,
                              inplace=True)
            
    elif agency == 'NHC': ############ NHC ###############
        data = parseNHC(filename)
        # change wind unit to m/s
        if wndunit == 'm/s':
            data['WND'].where(data['WND']==undef,
                              data['WND']*0.51444,
                              inplace=True)
            
    elif agency == 'IBTRACS': ######### IBTrACS ##########
        data = parseIBTrACS(filename)
        # change wind unit to m/s
        if wndunit == 'm/s':
            data['WND'].where(data['WND']==undef,
                              data['WND']*0.51444,
                              inplace=True)
        
    elif agency == 'BABJ': ############ BABJ #############
        data = parseBABJ(filename)
        # change wind unit to knot
        if wndunit == 'knot':
            data['WND'].where(data['WND']==undef,
                              data['WND']/0.51444,
                              inplace=True)
        
    else:
        raise Exception('not supported agent ' + agency + ', should be one of ' +
                        "['JTWC', 'JMA', 'CMA', 'NHC', 'IBTrACS', 'BABJ']")
    
    # remove duplicated records
    data.drop_duplicates(inplace=True)
    
    # change dtypes to save memory
    data = data.astype({'LAT': np.float32, 'LON': np.float32,
                        'WND': np.float32, 'PRS': np.float32})
    
    if 'IDtmp' in data:
        # unique for CMA, JTWC
        TCdata = data.groupby('IDtmp')
    else:
        # unique for NHC, JMA, IBTrACS
        TCdata = data.groupby('ID')
    
    TCs = []
    
    for groupID, DATA in TCdata:
        DATA.reset_index(drop=True, inplace=True)
        
        # data cleaning using condition
        if cond is not None:
            DATA = DATA.loc[cond]
            DATA.reset_index(drop=True, inplace=True)
            
            if len(DATA) == 0:
                continue
        
        year = DATA.iloc[0]['TIME'].year
        name = DATA['NAME'].iloc[0]
        ID   = DATA['ID'].iloc[0]
        
        if agency == 'BABJ': # operational
            for time, fcstdata in DATA.groupby('TIME'):
                tc = TC(ID, name, year, wndunit, time,
                        fcstdata.drop(columns=['ID', 'TIME', 'NAME'])
                             .reset_index(drop=True))
                TCs.append(tc)

        else: # Best Track
            drops = ['ID', 'NAME']
            
            if 'IDtmp' in DATA:
                drops.append('IDtmp')
                
            tc = TC(ID, name, year, wndunit, 0,
                    DATA.drop(columns=drops).reset_index(drop=True))
        
        TCs.append(tc)
    
    return TCSet(TCs, agency=agency)


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
                ID    = tokens[0].strip() # unique in NHC
                NAME  = tokens[1].strip()

                ID = ID[4:8] + ID[2:4]
                
                if NAME == 'UNNAMED':
                    NAME = 'NONAME'
                
                count = int(count)
                
                strIdx = i + 1
                
                for ln in fileContent[strIdx:strIdx+count]:
                    TIME = datetime.strptime(ln[:8] + ln[10:12], "%Y%m%d%H")
                    TYPE = __get_type_NHC(ln[19:21])
                    LAT  = float(ln[22:27].strip())
                    LON  = float(ln[30:35].strip())
                    PRS  = float(ln[43:47].strip())
                    WND  = float(ln[39:41].strip())

                    if ln[35:36] == 'W':
                        LON = 360 - LON
                    
                    if PRS == -999:
                        PRS = undef

                    if WND == -999:
                        WND = undef
                    
                    re.append((ID, NAME, TIME, LAT, LON, TYPE, PRS, WND))
    
    return pd.DataFrame.from_records(re, columns=['ID','NAME','TIME','LAT',
                                                  'LON','TYPE','PRS','WND'])


def parseIBTrACS(filename, encoding='utf-8'):
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
    keeps = ['sid', 'name', 'iso_time',
             # 'numobs', 'basin', 'season',
             'nature', 'lat', 'lon', 'wmo_wind', 'wmo_pres']
    
    KEEPS = [s.upper() for s in keeps]
    
    if filename.endswith('.nc'):
        ds = xr.open_dataset(filename)
        
        drops = [v for v in ds.variables]
        
        for keep in keeps:
            drops.remove(keep)
        
        ds = ds.drop_vars(drops)
        
        df = ds.to_dataframe().loc[lambda df:df['iso_time']!=b'']
        
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
        df.loc[:,'iso_time'] = df['iso_time'].str.decode('utf-8').astype(np.datetime64)
        df.loc[:,'sid'     ] = df['sid'     ].str.decode('utf-8')
        df.loc[:,'name'    ] = df['name'    ].str.decode('utf-8')
        df.loc[:,'nature'  ] = df['nature'  ].str.decode('utf-8')
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
                parse_dates=['ISO_TIME']).loc[lambda df:df['ISO_TIME']!='']
        
        df.replace(r'^\s+$', undef, regex=True, inplace=True)
        
    else:
        raise Exception('unsupported file type ' + filename)
    
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
                
                count  = parts[2].strip()
                IDtmp  = parts[3].strip() # this is unique in each year
                ID     = parts[4].strip() # official ID
                NAME   = parts[7].strip()
                count  = int(count)
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
                    LAT  = float(tokens[2]) / 10.0
                    LON  = float(tokens[3]) / 10.0
                    PRS  = float(tokens[4])
                    WND  = float(tokens[5]) # m/s
                    
                    re.append((IDtmp, ID, NAME, TIME, LAT, LON, TYPE, PRS, WND))
            
    return pd.DataFrame.from_records(re, columns=['IDtmp', 'ID', 'NAME',
                                                  'TIME', 'LAT', 'LON','TYPE',
                                                  'PRS', 'WND'])


def parseJMA(filename, encoding='utf-8'):
    """
    Parse the Best-Track data from Regional Specialized Meteorological
    Center (RSMC) Tokyo-Typhoon Center, Japan Meteorological Agency (JMA)
    into a pandas.DataFrame.

    Note that there are intensive observations to some recent TCs which are
    valid at 0300, 0900, 1500, 2100 UTCs.  These records can be filtered later.

    The maximum surface wind speed is defined as *10-min* averaged 10m wind
    speed in unit of *knot*.

    Reference:
    https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/Besttracks/e_format_bst.html

    Parameters
    ----------
    filename : str
        The file name of the JMA Best-Track data.
    encoding : str
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
            if line.startswith('66666'):
                count  = line[12:15].strip()
                ID     = line[21:25].strip() # unique
                IDtmp  = line[ 6:10].strip()
                NAME   = line[30:50].strip()
                count  = int(count)
                strIdx = i + 1
                
                if ID != IDtmp:
                    raise Exception(ID, IDtmp)
                
                tokens = fileContent[strIdx].split()
                year   = datetime.strptime(tokens[0],
                                           "%y%m%d%H").year
                
                # don't let year = 2069 happen
                if 51 <= int(tokens[0][:2]) <= 68:
                    year -= 100
                
                # add century to ID
                ID = str(year)[:2] + ID
                
                if NAME == '':
                    NAME = 'NONAME'
                
                for ln in fileContent[strIdx:strIdx+count]:
                    tokens = ln.split()
                    
                    TIME = datetime.strptime(tokens[0], "%y%m%d%H")
                    TYPE = __get_type_JMA(tokens[2])
                    LAT  = float(tokens[3]) / 10.0
                    LON  = float(tokens[4]) / 10.0
                    PRS  = float(tokens[5])
                    
                    # don't let year = 2069 happen
                    if 51 <= int(tokens[0][:2]) <= 68:
                        TIME = TIME.replace(year=TIME.year-100)
                    
                    if len(tokens) <= 6:
                        WND = undef
                    else:
                        WND = float(tokens[6])
                    
                    re.append((ID, NAME, TIME, LAT, LON, TYPE, PRS, WND))
    
    return pd.DataFrame.from_records(re, columns=['ID','NAME','TIME','LAT',
                                                  'LON','TYPE','PRS','WND'])


def parseJTWC(filenames, encoding='utf-8'):
    """
    Parse the Best-Track data from Joint Typhoon Warning Center (JTWC)
    into a pandas.DataFrame.
    
    Note that there is no official ID and name for each TC.  So the ID is
    a simple composite of year + number of TC, while the name is the
    composite of basin (e.g., WP) + ID.  Also there are some duplicated data.
    
    The maximum surface wind speed is defined as *1-min* averaged 10m wind
    speed in unit of *knot*.
    
    Possible errors in the original txt files:
    - bwp231969.txt, time 1969100000 should be 1969100100?
    
    Reference: https://www.metoc.navy.mil/jtwc/jtwc.html?best-tracks
    
    Parameters
    ----------
    filenames : str
        The file name(s) of the JTWC Best-Track data.
    encoding : str
        Encoding of the file.

    Returns
    -------
    re: pandas.DataFrame
        Raw data in a pandas.DataFrame.
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
        
        IDtmp = os.path.splitext(os.path.basename(p))[0]
        
        if fileContent != []:
            pre = fileContent[0][10:12]
            ID  = pre + fileContent[0][4:6]
            
            if int(pre) >= 45:
                ID = '19' + ID
            else:
                ID = '20' + ID
            
            for ln in fileContent:
                # basin= ln[:2]
                TIME = datetime.strptime(ln[8:18], "%Y%m%d%H")
                LAT  = float(ln[35:38]) / 10.0
                LON  = float(ln[41:45]) / 10.0
                WND  = float(ln[47:51])
                
                if ln[38:39] == 'S':
                    LAT = -LAT
                
                if ln[45:46] == 'W':
                    LON = 360.0 - LON
                
                tmp = ln[52:57].strip()
                if tmp == '':
                    PRS = undef
                else:
                    PRS = float(tmp)
                
                if WND == '-999':
                    WND = undef
                
                re.append((IDtmp, ID, 'NONAME', TIME, LAT, LON, PRS, WND))
        else:
            print('empty file is found: ' + p)
    
    return pd.DataFrame.from_records(re, columns=['IDtmp', 'ID', 'NAME',
                                                  'TIME', 'LAT', 'LON',
                                                  'PRS', 'WND'])


def parseBABJ(filenames, encoding='GBK'):
    """
    Parse China Meteorological Administration (CMA) operational forecast data
    (babj format) into a pandas.DataFrame.

    Note that the data use Beijing time and they are changed to UTC during the
    parsing.  We also remove those records with pressure data undefined.

    Parameters
    ----------
    filenames: str
        The file name of the BABJ data.
    encoding : str
        Encoding of the file.

    Returns
    -------
    re: pandas.DataFrame
        Raw data in a pandas.DataFrame.
    """
    paths = sorted(glob(filenames))
    
    re = []
    
    for path in paths:
        with open(path, 'r', encoding=encoding) as f:
            fileContent = f.readlines()
            
            tokens = fileContent[1].split()
            
            NAME = tokens[0]
            ID   = tokens[1]
            count= tokens[3]
            
            count = int(count)
            
            strIdx = 3
            
            for ln in fileContent[strIdx:strIdx+count]:
                tokens = ln.split()
                
                timestr = tokens[0] + tokens[1] + tokens[2] + tokens[3]
                
                # change BJ time to UTC
                TIME = datetime.strptime(timestr, "%Y%m%d%H") \
                     - timedelta(hours=8)
                FCST =   int(tokens[4])
                LON  = float(tokens[5])
                LAT  = float(tokens[6])
                PRS  = float(tokens[7])
                WND  = float(tokens[8])
                
                if PRS != 9999.0:
                    re.append((ID, NAME, TIME, FCST, LAT, LON, PRS, WND))
                else:
                    print('remove undef in ' + ID + ':\n' + ln.strip())
    
    return pd.DataFrame.from_records(re, columns=['ID', 'NAME', 'TIME', 'FCST',
                                                  'LAT', 'LON', 'PRS', 'WND'])



"""
Helper (private) methods are defined below
"""
def __concat_files(paths, encoding='utf-8'):
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
        raise OSError("no files to open")

    re = []

    for p in paths:
        with open(p, 'r', encoding=encoding) as f:
            content = f.readlines()

            for line in content:
                if line.find('\n') != -1:
                    re.append(line)
                else:
                    re.append(line+'\n')
    
    return re


def __get_type_JMA(code):
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
    if code in ['1', '7', '8', '9']:
        return 'OTHERS'
    elif code == '2':
        return 'TD'
    elif code == '3':
        return 'TS'
    elif code == '4':
        return 'STS'
    elif code == '5':
        return 'TY'
    elif code == '6':
        return 'EC'
    else:
        raise Exception('unknown code ' + code)


def __get_type_CMA(code):
    """
    Get the intensity category according to "Chinese National Standard for
    Grade of Tropical Cyclones", which was put in practice since 15 June 2006.
    
    Reference:
    http://tcdata.typhoon.org.cn/zjljsjj_sm.html
    
    0 - weaker than TD or unknown
	1 - tropical depression	  (TD , 10.8-17.1 m/s)
	2 - tropical storm	      (TS , 17.2-24.4 m/s)
	3 - severe tropical storm (STS, 24.5-32.6 m/s)
	4 - typhoon				  (TY , 41.4-32.7 m/s)
	5 - severe typhoon		  (STY, 41.5-50.9 m/s)
	6 - super typhoon		  (superTY, >=51.0 m/s)
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
    if   code == '0':
        return 'OTHERS'
    elif code == '1':
        return 'TD'
    elif code == '2':
        return 'TS'
    elif code == '3':
        return 'STS'
    elif code == '4':
        return 'TY'
    elif code == '5':
        return 'STY'
    elif code == '6':
        return 'STY'
    elif code == '9':
        return 'EC'
    else:
        raise Exception('unknown code ' + code)


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



"""
Test codes
"""
if __name__ == '__main__':
    print('ok')

    re = __concat_files('D:/Data/Typhoons/CMA/original/CH1949BST.txt')

    print(re)

