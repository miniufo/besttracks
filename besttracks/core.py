# -*- coding: utf-8 -*-
"""
Created on 2020.08.01

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
import pandas as pd

undef = -9999.0


"""
Data structs for Best-Track data (TC) and forecast data (TCfcst)
"""

class Particle(object):
    """
    This class represents a single Lagrangian particle.
    """
    def __init__(self, ID, records):
        """
        Constructor.
        
        Parameters
        ----------
        ID: str
            Unique ID of the particle.
        records: pandas.DataFrame
            Records of the particle.

        Returns
        ----------
        p: Particle
            A single Lagrangian particle
        """
        self.ID = ID
        self.records = records
    
    
    def sel(self, cond, copy=True):
        """
        Select records meet a given condition.  Similar to DataFrame.sel().
        
        Parameters
        ----------
        cond: lambda expression
            Condition that records should meet e.g.,:
            1. Choose standard UTC times
               cond = lambda df: df['TIME'].dt.hour.isin([0, 6, 12, 18])
            2. Choose records in year 2010
               cond = lambda df: df['TIME'].dt.year == 2010
            3. Choose records with wind above 30
               cond = lambda df: df['WND'] > 30
            4. Drop records with wind being 0
               cond = lambda df: df['WND'] != 0
            5. Choose records inside a region:
               cond = lambda df: (df['LON'] > 90 & df['LON'] < 180 &
                                  df['LAT'] >  0 & df['LAT'] < 50)
            6. Choose records by slicing:
               cond = slice(2, 10)
        
        Returns
        ----------
        re: pandas.DataFrame
            A new particle containing the records satisfying the conditions
        """
        re = self.copy(copy_records=False)
        
        re.records = self.records.loc[cond].reset_index(drop=True)
        
        return re
    
    
    def translate_velocity(self, Rearth=6371200):
        """
        Calculate translate velocity of this TC.

        Parameters
        ----------
        Rearth: float
            Mean earth radius (meter)
        
        Returns
        ----------
        re: pandas.DataFrame
            A new particle containing the records satisfying the conditions
        """
        from numpy import cos as cos
        from numpy import deg2rad as deg2rad
        
        lons = self.records.LON
        lats = self.records.LAT
        time = self.records.TIME
        
        dlon = finite_difference(lons)
        dlat = finite_difference(lats)
        dtim = finite_difference(time)[0].seconds
        
        uo = Rearth * deg2rad(dlon * cos(deg2rad(lats))) / dtim
        vo = Rearth * deg2rad(dlat) / dtim
        
        self.records['UO'] = uo
        self.records['VO'] = vo
        
        return self
    
    
    def get_as_xarray(self, field):
        """
        Return a field of records as a xarray of time series.
        
        Parameters
        ----------
        field: str
            A field string of the records.
        
        Returns
        ----------
        re: xarray.DataArray
            A time series of the given field in xarray
        """
        import xarray as xr
        
        return xr.DataArray(self.records[field.upper()], dims='time',
                            coords={'time':self.records['TIME']})
    
    
    def duration(self):
        """
        Get duration (days) of this particle.

        Returns
        ----------
        re: float
            Duration of this particle in unit of day
        """
        duration = np.ptp(self.records['TIME'])
        return pd.to_timedelta([duration]).astype('timedelta64[h]')[0] / 24.0
    
    
    def resample(self, *args, **kwargs):
        """
        Re-sample a particle along time dimension.
        
        Parameters
        ----------
        args: list
            All the positional arguments passing to `DataFrame.resample()`.
        kwargs: list
            All the keyword arguments passing to `DataFrame.resample()`.

        Returns
        ----------
        re: particle
            Resampled particle
        """
        re = self.copy(copy_records=False)
        re.records = self.records.set_index('TIME').resample(*args, **kwargs)\
                         .interpolate().fillna(value=None,
                                               method='ffill').reset_index()
        
        return re
    
    
    def plot_track(self, **kwargs):
        """
        Plot the track of this particle.
        """
        from .utils import plot_track
        return plot_track(self, **kwargs)
    
    def binning(self, var=None, **kwargs):
        """
        Plot the track and intensity of the TC.
        """
        from .utils import binning_particle
        return binning_particle(self, var=var, **kwargs)
        
    
    def copy(self, copy_records=True):
        """
        Make a copy of the current particle.

        Parameters
        ----------
        copy_records: bool
            Copy the records or not.

        Returns
        ----------
        re: particle
            A copy of this particle.
        """
        s = self
        
        if copy_records:
            re = type(self)(s.ID, s.records.copy())
        else:
            re = type(self)(s.ID, None)
        
        return re
    
    
    def __len__(self):
        """
        Get the number of records.
        """
        return self.records.__len__()
    
    def __iter__(self):
        """
        Iterator.
        """
        return self.records.itertuples()
    
    
    def __getitem__(self, key):
        """
        Used to iterate over the particle record.
        """
        if isinstance(key, int):
            return self.records.iloc[[key]]
        elif isinstance(key, str):
            if key in ['ID']:
                return self.__dict__[key]
            else:
                return self.records[key]
        else:
            raise Exception('invalid type of key, should be int or str')
    
    def __repr__(self):
        """
        Used to print the Drifter.
        """
        s = self
        
        info = ('Particle (ID={0:8s})\n').format(str(s.ID))
        
        return info + s.records.__repr__()


class TC(Particle):
    """
    This class represents a single tropical cyclone (TC).
    """
    def __init__(self, ID, name, year, wndunit, fcstTime, records):
        """
        Constructor.
        
        Parameters
        ----------
        ID: str
            Official ID of the TC.
        name: str
            Official name of the TC.
        year: str
            Year of the TC.
        wndunit: str
            Unit of the wind, ['knot', 'm/s'].
        fcstTime: int
            Valid forecast time in hours e.g. (0, 24, 48...).
        records: pandas.DataFrame
            Records of the TC.

        Returns
        ----------
        TC: TC
            A single TC
        """
        super().__init__(ID, records)
        
        self.name = name
        self.year = year
        self.wndunit = wndunit.lower()
        self.fcstTime = fcstTime
        
        if wndunit not in ['knot', 'm/s']:
            raise Exception('invalid wind unit, should be "knot" or "m/s"')
    
    def change_wind_unit(self, unit=None):
        """
        Change (in-place) the wind unit between knot and m/s.
        
        Parameters
        ----------
        unit: str
            Either knot or m/s.
        """
        recs = self.records
        
        if unit is None:
            if self.wndunit == 'knot':
                recs['WND'].where(recs['WND']==undef,
                                  recs['WND'] * 0.51444, # to m/s
                                  inplace=True)
                self.wndunit = 'm/s'
            else:
                recs['WND'].where(recs['WND']==undef,
                                  recs['WND'] / 0.51444, # to knot
                                  inplace=True)
                self.wndunit = 'knot'
        else:
            if unit != self.wndunit:
                if self.wndunit == 'knot':
                    recs['WND'].where(recs['WND']==undef,
                                      recs['WND'] * 0.51444, # to m/s
                                      inplace=True)
                    self.wndunit = 'm/s'
                else:
                    recs['WND'].where(recs['WND']==undef,
                                      recs['WND'] / 0.51444, # to knot
                                      inplace=True)
                    self.wndunit = 'knot'
        
        return self
        
    
    def ace(self):
        """
        Get accumulated cyclone energy (ACE) of this TC.
        """
        cond = lambda df: df['TIME'].dt.hour.isin([0, 6, 12, 18])
        
        wnd = self.records.loc[cond]['WND']
        
        wnd = wnd.where(wnd!=undef)
        
        return (wnd*wnd).sum()
        
    
    def copy(self, copy_records=True):
        """
        Make a copy of the current TC.

        Parameters
        ----------
        copy_records: bool
            Copy the records or not.

        Returns
        ----------
        re: TC
            A copy of this TC.
        """
        s = self
        
        if copy_records:
            re = type(self)(s.ID, s.name, s.year, s.wndunit,
                    s.fcstTime, s.records.copy())
        else:
            re = type(self)(s.ID, s.name, s.year, s.wndunit,
                    s.fcstTime, None)
        
        return re
    
    def peak_intensity(self):
        """
        Get the peak intensity given PRS or WND.
        
        Returns
        ----------
        re: tuple
            Peak intensity of pressure and wind
        """
        prs = self.records['PRS']
        wnd = self.records['WND']
        
        minP, Ppos = prs.min(), prs.argmin()
        maxW, Wpos = wnd.max(), wnd.argmax()
        
        if minP != undef and maxW != undef:
            if Ppos != Wpos:
                if prs[Wpos] == minP:
                    return minP, maxW
                elif wnd[Ppos] == maxW:
                    return minP, maxW
                else:
                    return min(prs[wnd==maxW]), maxW
        
        return minP, maxW
    
    def plot_intensity(self, unit='knot', **kwargs):
        """
        Plot the intensity of the TC.
        """
        from .utils import plot_intensity
        return plot_intensity(self, **kwargs)
    
    def plot(self, **kwargs):
        """
        Plot the track and intensity of the TC.
        """
        from .utils import plot
        plot(self, **kwargs)
    
    def plot_track(self, **kwargs):
        """
        Plot the track and intensity of the TC.
        """
        from .utils import plot_track
        return plot_track(self, add_legend=True, **kwargs)
    
    def __repr__(self):
        """
        Used to print the TC.
        """
        s = self
        
        info = ('TC (ID={0:s}, name={1:s}, year={2:4d}, ' +
               'fcstTime={3:s}, unit={4:s})\n') \
                .format(s.ID, s.name, s.year, str(s.fcstTime), s.wndunit)
        
        return info + s.records.__repr__()


class Drifter(Particle):
    """
    This class represents a single tropical cyclone (TC).
    """
    def __init__(self, ID, records):
        """
        Constructor.
        
        Parameters
        ----------
        ID: str
            Official ID of the drifter.
        records: pandas.DataFrame
            Records of the drifter.

        Returns
        ----------
        re: drifter
            A single drifter
        """
        super().__init__(ID, records)
    
    
    def __repr__(self):
        """
        Used to print the Drifter.
        """
        s = self
        
        info = ('Drifter (ID={0:8s})\n').format(s.ID)
        
        return info + s.records.__repr__()



class ParticleSet(object):
    """
    This class represents a set of Particles.
    """
    def __init__(self, particles):
        """
        Constructor.
        
        Parameters
        ----------
        particles: list
            A list of particles.

        Returns
        ----------
        re : ParticleSet
            A set of particles
        """
        self.particles = particles
    
    
    def select(self, cond):
        """
        Select drifter(s) given a condition.
        
        Parameters
        ----------
        cond: lambda expression
            Whether a drifter meets the condition.
        """
        ps = list(filter(cond, self.particles))
        
        if ps:
            return type(self)(ps)
        else:
            print('no particles are found')
    
    
    def groupby(self, field):
        """
        Group the Particles into different categraries according to field.
        
        Parameters
        ----------
        field: str
            field in ['ID'].
        """
        from itertools import groupby
        
        gs = groupby(self.particles, key=lambda p:p.__getattribute__(field))
        
        flds = []
        sets = []
        
        for f, ps in gs:
            flds.append(f)
            sets.append(type(self)([p for p in ps]))
        
        return zip(flds, sets)
    
    
    def total_duration(self):
        """
        Get total duration of this ParticleSet in unit of days.
        """
        return sum([p.duration() for p in self.particles])
    
    def plot_tracks(self, **kwargs):
        from .utils import plot_tracks
        return plot_tracks(self, **kwargs)
    
    def binning(self, var=None, **kwargs):
        """
        Binning the Eulerian track statistics of this set.
        """
        from .utils import binning_particles
        return binning_particles(self, var=var, **kwargs)
    
    
    def __len__(self):
        """
        Get the number of particles.
        """
        return self.particles.__len__()
    
    def __getitem__(self, key):
        """
        Used to iterate over the whole ParticleSet.
        """
        return self.particles[key]
    
    def __repr__(self):
        """
        Used to print the TC dataset.
        """
        info = []
        
        info.append('Particle dataset:\n')
        info.append('  {0:2d} particles, {1:6.1f} particle-days\n'
                    .format(len(self.particles), self.total_duration()))
        
        minPa, maxPa = self._find_duration_extrema()
        
        info.append('  longest  drifter {0:8s}: {1:5.2f} days \n'.format(
                    maxPa.ID, maxPa.duration()))
        info.append('  shortest drifter {0:8s}: {1:5.2f} days \n'.format(
                    minPa.ID, minPa.duration()))
        
        return ''.join(info)
    
    def _find_duration_extrema(self):
        """
        Find minimum and maximum durations of particles.
        
        Returns
        ----------
        minPa, maxPa: tuple
            Minimum and maximum duration particles
        """
        minDr, maxDr = -undef, undef
        minPa, maxPa =  None , None
        
        for ps in self.particles:
            duration = ps.duration()
            
            if duration < minDr:
                minDr = duration
                minPa = ps
            
            if duration > maxDr:
                maxDr = duration
                maxPa = ps
        
        return minPa, maxPa

class TCSet(ParticleSet):
    """
    This class represents a set of tropical cyclones (TCs).
    """
    def __init__(self, TCs, agency=None):
        """
        Constructor.
        
        Parameters
        ----------
        TCs: list
            A list of TCs.
        agency: str
            agency that provide the data e.g., ['CMA', 'JMA', 'JTWC',
                                                'NHC', 'IBTrACS']

        Returns
        ----------
        TCs : TCSet
            A set of TCs
        """
        super().__init__(TCs)
        
        self.agency = agency
    
    def ace(self):
        """
        Get accumulated cyclone energy (ACE) of the whole dataset.
        """
        return sum([tc.ace() for tc in self.particles])
    
    def plot_intensities(self, unit='knot', **kwargs):
        """
        Plot the intensities of the TCSet.  This is useful for plotting the
        ensemble forecasts of a single TC initialized at different times.
        """
        from .utils import plot_intensities
        return plot_intensities(self, **kwargs)
    
    def change_wind_unit(self, unit=None):
        """
        Change (in-place) the wind unit between knot and m/s.
        
        Parameters
        ----------
        unit: str
            Either knot or m/s.
        """
        for tc in self.particles:
            tc.change_wind_unit(unit=unit)
        
            # if unit != None:
            #     tc.wndunit = unit
            # else:
            #     if tc.wndunit == 'knot':
            #         tc.wndunit = 'm/s'
            #     else:
            #         tc.wndunit = 'knot'
        
        return self
    
    def plot_timeseries(self, freq='monthly', **kwargs):
        """
        Plot the timeseries of statistics.
        
        Parameters
        ----------
        freq: str
            Frequency of statistics, should be ['monthly', 'yearly', 'annual'].
        """
        from .utils import plot_timeseries
        return plot_timeseries(self, freq=freq, **kwargs)
    
    
    def __repr__(self):
        """
        Used to print the TC dataset.
        """
        if len(self.particles) == 0:
            return 'No TCs in this TCSet'
        
        info = []
        
        if self.agency == None:
            info.append('TC best-track dataset:\n')
        else:
            info.append('TC best-track dataset ({0:s}):\n'.format(self.agency))
        
        yrs = [p.year for p in self.particles]
        
        info.append('  {0:1d} TCs from {1:4d} to {2:4d}, {3:6.1f} cyclone days\n'
                    .format(len(self.particles), min(yrs),
                            max(yrs), self.total_duration()))
        
        minTC, maxTC = self._find_duration_extrema()
        
        info.append('  longest   TC {0:s}, {1:9s}: {2:5.2f} days \n'.format(
                    maxTC.ID, maxTC.name, maxTC.duration()))
        info.append('  shortest  TC {0:s}, {1:9s}: {2:5.2f} days \n'.format(
                    minTC.ID, minTC.name, minTC.duration()))
        
        minTC, maxTC = self.__find_intensity_extrema()
        
        minP, maxW = maxTC.peak_intensity()
        unit = maxTC.wndunit
        info.append('  strongest TC {0:s}, {1:9s}: {2:5.0f} hPa, {3:5.1f} {4:s}\n'
                    .format(maxTC.ID, maxTC.name, minP, maxW, unit))
        
        minP, maxW = minTC.peak_intensity()
        unit = maxTC.wndunit
        info.append('  weakest   TC {0:s}, {1:9s}: {2:5.0f} hPa, {3:5.1f} {4:s}\n'
                    .format(minTC.ID, minTC.name, minP, maxW, unit))
        
        minTC, maxTC = self.__find_ace_extrema()
        
        maxA = maxTC.ace()
        unit = maxTC.wndunit
        info.append('  largest  ACE {0:s}, {1:9s}: {2:5.0f} ({3:s})^2\n'
                    .format(maxTC.ID, maxTC.name, maxA, unit))
        
        minA = minTC.ace()
        unit = maxTC.wndunit
        info.append('  smallest ACE {0:s}, {1:9s}: {2:5.0f} ({3:s})^2\n'
                    .format(minTC.ID, minTC.name, minA, unit))
        
        return ''.join(info)
    
    def __find_intensity_extrema(self):
        """
        Find strongest and weakest TCs.
        
        Returns
        ----------
        minTC, maxTC: tuple
            Weakest and strongest TCs
        """
        minWd, maxWd = -undef, undef
        minPr, maxPr = -undef, undef
        minTC, maxTC =  None,  None
        
        for tc in self.particles:
            minP, maxW = tc.peak_intensity()
            
            if maxW != undef:
                if maxW > maxWd:
                    maxWd = maxW
                    maxTC = tc
                
                if maxW < minWd:
                    minWd = maxW
                    minTC = tc
                    
            else:
                if minP < minPr:
                    minPr = minP
                    maxTC = tc
                
                if minP > maxPr:
                    maxPr = minP
                    minTC = tc
        
        return minTC, maxTC
    
    def __find_ace_extrema(self):
        """
        Find minimum and maximum ACE of TCs.
        
        Returns
        ----------
        minTC, maxTC: tuple
            Minimum and maximum duration TCs
        """
        minDr, maxDr = -undef, undef
        minTC, maxTC =  None , None
        
        for tc in self.particles:
            ace = tc.ace()
            
            if ace < minDr or minDr == -undef:
                minDr = ace
                minTC = tc
                
            if ace > maxDr or maxDr == undef:
                maxDr = ace
                maxTC = tc
        
        return minTC, maxTC

class DrifterSet(ParticleSet):
    """
    This class represents a set of surface drifters.
    """
    def __init__(self, drifters):
        """
        Constructor.
        
        Parameters
        ----------
        drifters: list
            A list of drifters.

        Returns
        ----------
        re : DrifterSet
            A set of drifters
        """
        super().__init__(drifters)
    
    
    def __repr__(self):
        """
        Used to print the drifter dataset.
        """
        info = []
        
        info.append('drifter dataset:\n')
        info.append('  {0:2d} drifters, {1:6.1f} drifter-days\n'
                    .format(len(self.particles), self.total_duration()))
        
        minPa, maxPa = self._find_duration_extrema()
        
        info.append('  longest  drifter {0:8s}: {1:5.2f} days \n'.format(
                    str(maxPa.ID), maxPa.duration()))
        info.append('  shortest drifter {0:8s}: {1:5.2f} days \n'.format(
                    str(minPa.ID), minPa.duration()))
        
        return ''.join(info)


"""
Helper (private) methods are defined below
"""
def finite_difference(array):
    """
    Central finite difference of a given data series.
    Forword or backword differences are used at the end points.
    
    Returns
    ----------
    re: dataframe or numpy.array
        Difference of original array.
    """
    if type(array) in [pd.core.series.Series, pd.core.frame.DataFrame]:
        dataL = array.shift( 1, fill_value=array.iloc[ 0])
        dataR = array.shift(-1, fill_value=array.iloc[-1])
    elif type(array) in [np.ndarray, np.array]:
        dataL = array.shift( 1, fill_value=array.iloc[ 0])
        dataR = array.shift(-1, fill_value=array.iloc[-1])
    else:
        raise Exception('invalid type of input: ' + type(array))
    
    # interior is central finite difference
    de = np.ones(len(array))
    de[1:-1] = 2
    
    re = (dataR - dataL) / de
    
    return re
    
    

"""
Test codes
"""
if __name__ == '__main__':
    issubclass(type(TCSet), ParticleSet)

