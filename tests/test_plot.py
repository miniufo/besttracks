# -*- coding: utf-8 -*-
"""
Created on 2020.08.17

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%% load data
import sys
sys.path.append('e:/OneDrive/Python/MyPack/')
from besttracks.besttracks import parse_TCs


cond = lambda df: df['TIME'].dt.hour.isin([0, 6, 12, 18])
# cond = lambda df: df['TIME'].dt.year==1958

def choose_records(df, windthreshold):
    strIdx = (df['WND'].values       >= windthreshold).argmax()
    endIdx = (df['WND'].values[::-1] >= windthreshold).argmax()
    
    endIdx = len(df) - endIdx - 1
    
    # print(strIdx, endIdx)
    
    return slice(strIdx, endIdx, None)

cond2 = lambda df: choose_records(df, 34)


TCs_ibtracs = parse_TCs('d:/Data/Typhoons/IBTrACS/IBTrACS.ALL.v04r00.nc',
                rec_cond=cond,
                agency='IBTrACS')

TCs_jma = parse_TCs('d:/Data/Typhoons/JMA/original/bst_all.txt',
                rec_cond=cond,
                agency='JMA')

TCs_cma = parse_TCs('d:/Data/Typhoons/CMA/original/*.txt',
                rec_cond=cond,
                agency='CMA')

TCs_jtwc1 = parse_TCs('d:/Data/Typhoons/JTWC/original/bwp/bwp*',
                rec_cond=cond,
                agency='JTWC')

TCs_jtwc2 = parse_TCs('d:/Data/Typhoons/JTWC/original/bsh/bsh*',
                rec_cond=cond,
                agency='JTWC')

TCs_jtwc3 = parse_TCs('d:/Data/Typhoons/JTWC/original/bio/bio*',
                rec_cond=cond,
                agency='JTWC')

TCs_nhc1 = parse_TCs('d:/Data/Typhoons/NHC/original/hurdat2-1851-2019-052520.txt',
                rec_cond=cond,
                agency='NHC')

TCs_nhc2 = parse_TCs('d:/Data/Typhoons/NHC/original/hurdat2-nepac-1949-2019-042320.txt',
                rec_cond=cond,
                agency='NHC')

print(TCs_ibtracs)
print(TCs_jma)
print(TCs_cma)
print(TCs_jtwc1)
print(TCs_jtwc2)
print(TCs_jtwc3)
print(TCs_nhc1)
print(TCs_nhc2)

#%%
TCs_ibtracs.plot_tracks(trackonly=True, linewidth=0.8, figsize=(15,8), add_legend=False)
TCs_jma.plot_tracks(trackonly=False, linewidth=0.8, figsize=(15,8))
TCs_cma.plot_tracks(trackonly=False, linewidth=0.8, figsize=(15,8))
TCs_jtwc1.plot_tracks(trackonly=True, linewidth=0.8, figsize=(15,8))
TCs_jtwc2.plot_tracks(trackonly=True, linewidth=0.8, figsize=(15,8))
TCs_jtwc3.plot_tracks(trackonly=True, linewidth=0.8, figsize=(15,8))
TCs_nhc1.plot_tracks(trackonly=False, linewidth=0.8, figsize=(15,8))
TCs_nhc2.plot_tracks(trackonly=False, linewidth=0.8, figsize=(15,8))


#%%
TCs_cma.binning_TCs()
TCs_jma.binning_TCs()
TCs_jtwc1.binning_TCs()
TCs_jtwc2.binning_TCs()
TCs_jtwc3.binning_TCs()
TCs_nhc1.binning_TCs()
TCs_nhc2.binning_TCs()
TCs_ibtracs.binning_TCs()

#%%
TCs_cma.binning_TCs('WND')
TCs_jma.binning_TCs('WND')
TCs_jtwc1.binning_TCs('WND')
TCs_jtwc2.binning_TCs('WND')
TCs_jtwc3.binning_TCs('WND')
TCs_nhc1.binning_TCs('WND')
TCs_nhc2.binning_TCs('WND')
TCs_ibtracs.binning_TCs('WND')

