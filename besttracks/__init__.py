# -*- coding: utf-8 -*-
"""
besttracks: a unified interface to tropical cyclone best-track datasets.

Provides parsers for JTWC, CMA, JMA, NHC, IBTrACS, and GDP drifter datasets,
along with plotting and binning utilities for TC best-track analyses.
"""
from .io import parse_TCs, parseBABJ, parseCMA, parseJMA, parseJTWC, \
                parseIBTrACS, parseNHC, parse_GDPDrifters
from .core import Particle, TC, Drifter, ParticleSet, TCSet, DrifterSet
from .utils import plot_tracks, plot_track, plot_intensity, plot_intensities, \
                   plot, binning, binning_particle, binning_particles

__version__ = "0.2.1"
