# -*- coding: utf-8 -*-
"""
Created on 2020.08.17

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
import numpy as np
import xarray as xr
import sys
import pandas as pd
sys.path.append('e:/OneDrive/Python/MyPack/')


#%% JTWC
from besttracks.besttracks import parseJTWC

agency = 'JTWC'
cond = lambda df:df['TIME'].dt.hour.isin([0, 6, 12, 18])

data = parseJTWC('d:/Data/Typhoons/'+agency+'/original/bwp/bwp*')

print(data)


#%% JMA
from besttracks.besttracks import parseJMA

agency = 'JMA'
cond = lambda df:df['TIME'].dt.hour.isin([0, 6, 12, 18])

data = parseJMA('d:/Data/Typhoons/'+agency+'/original/bst_all.txt')

print(data)


#%% CMA
from besttracks.besttracks import parseCMA

agency = 'CMA'
cond = lambda df:df['TIME'].dt.hour.isin([0, 6, 12, 18])

data = parseCMA('d:/Data/Typhoons/'+agency+'/original/*.txt')

print(data)


#%% NHC
from besttracks.besttracks import parseNHC

agency = 'NHC'
cond = lambda df:df['TIME'].dt.hour.isin([0, 6, 12, 18])

data = parseNHC('d:/Data/Typhoons/'+agency+'/original/bwp/bwp*')

print(data)


#%% IBTrACS
from besttracks.besttracks import parseIBTrACS, parse_TCs

agency = 'IBTrACS'
cond = lambda df:~df['TIME'].dt.hour.isin([0, 6, 12, 18])

# df = parseIBTrACS('d:/Data/Typhoons/'+agency+'/ibtracs.ALL.list.v04r00.csv')
# df = parseIBTrACS('d:/Data/Typhoons/'+agency+'/IBTrACS.ALL.v04r00.nc')

# print(df)

TCs_ibtracs1 = parse_TCs('d:/Data/Typhoons/'+agency+'/ibtracs.ALL.list.v04r00.csv',
                cond=cond,
                agency='IBTrACS')
TCs_ibtracs2 = parse_TCs('d:/Data/Typhoons/'+agency+'/IBTrACS.ALL.v04r00.nc',
                cond=cond,
                agency='IBTrACS')


#%% BABJ
from besttracks.besttracks import parseBABJ

agency = 'BABJ'
cond = lambda df:df['TIME'].dt.hour.isin([0, 6, 12, 18])

data = parseBABJ('d:/Data/Typhoons/'+agency+'/babj1601.dat')

print(data)


# %%
data_sel = data[#(data['NAME']=='AMY') &
                (data['ID']=='193204')
                # (data['TIME'].dt.year==1977)
                ].dropna()

print(data_sel)


# %%
