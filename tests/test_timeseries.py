# -*- coding: utf-8 -*-
"""
Created on 2020.08.17

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
import pandas as pd
from besttracks.besttracks import Particle, TC, Drifter, \
        ParticleSet, TCSet, DrifterSet, parse_TCs

cond_stdUTC = lambda df: ((df['TIME'].dt.hour.isin([0, 6, 12, 18])) &
                          (df['WND'] > 10))

tcs = parse_TCs('d:/Data/Typhoons/JTWC/original/bwp/bwp*.txt',
               agency='JTWC',
               rec_cond=cond_stdUTC)

tcs.plot_timeseries(freq='monthly')
tcs.plot_timeseries(freq='yearly')
tcs.plot_timeseries(freq='annual')

