# -*- coding: utf-8 -*-
"""
Created on 2020.08.17

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
#%%
from besttracks.besttracks import Particle, TC, Drifter, \
        ParticleSet, TCSet, DrifterSet, parse_TCs

tcs = parse_TCs('d:/Data/Typhoons/JTWC/original/bwp/bwp*.txt',
               agency='JTWC')

# tcs.plot_tracks()

#%%
tcs = parse_TCs('H:/Data/ULFI/babj/BT/2016/babj*.dat',
               agency='babj')

# tcs.plot_tracks()

