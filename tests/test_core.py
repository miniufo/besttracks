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
from BestTracks.BestTracks import parseJTWC

agent = 'JTWC'
cond = lambda df:df['TIME'].dt.hour.isin([0, 6, 12, 18])

data = parseJTWC('d:/Data/Typhoons/'+agent+'/original/bwp/bwp*')

print(data)


#%% JMA
from BestTracks.BestTracks import parseJMA

agent = 'JMA'
cond = lambda df:df['TIME'].dt.hour.isin([0, 6, 12, 18])

data = parseJMA('d:/Data/Typhoons/'+agent+'/original/bst_all.txt')

print(data)


#%% CMA
from BestTracks.BestTracks import parseCMA

agent = 'CMA'
cond = lambda df:df['TIME'].dt.hour.isin([0, 6, 12, 18])

data = parseCMA('d:/Data/Typhoons/'+agent+'/original/*.txt')

print(data)


#%% NHC
from BestTracks.BestTracks import parseNHC

agent = 'NHC'
cond = lambda df:df['TIME'].dt.hour.isin([0, 6, 12, 18])

data = parseNHC('d:/Data/Typhoons/'+agent+'/original/bwp/bwp*')

print(data)


#%% BABJ
from BestTracks.BestTracks import parseBABJ

agent = 'BABJ'
cond = lambda df:df['TIME'].dt.hour.isin([0, 6, 12, 18])

data = parseBABJ('d:/Data/Typhoons/'+agent+'/babj1601.dat')

print(data)


# %%
data_sel = data[#(data['NAME']=='AMY') &
                (data['ID']=='193204')
                # (data['TIME'].dt.year==1977)
                ].dropna()

print(data_sel)


# %%
