# -*- coding: utf-8 -*-
"""
Created on 2020.08.01

@author: MiniUFO
Copyright 2018. All rights reserved. Use is subject to license terms.
"""
import numpy as np
from .utils import plot_track, plot_intensity, plot

undef = -9999.0


"""
Data structs for Best-Track data (TC) and forecast data (TCfcst)
"""
# Storm = namedtuple('Storm', ['ID', 'name', 'year', 'fcstTime', 'data'])

# TCType = np.dtype([('ID'      , np.str_, 6),
#                    ('name'    , np.str_, 15),
#                    ('year'    , np.int32),
#                    ('fcstTime', np.int16),
#                    ('data'    , pd.DataFrame)])


class TCSet(object):
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
        self.TCs    = TCs
        self.agency = agency
    
    def plot_tracks(self, **kwargs):
        from besttracks.besttracks import plot_tracks
        return plot_tracks(self, **kwargs)
    
    def total_cyclone_days(self):
        """
        Get total cyclone days of this TCSet.
        """
        return sum([tc.duration() for tc in self.TCs])
    
    def change_wind_unit(self, unit=None):
        """
        Change (in-place) the wind unit between knot and m/s.
        
        Parameters
        ----------
        unit: str
            Either knot or m/s.
        """
        for tc in self.TCs:
            tc.change_wind_unit(unit=unit)
        
            if unit != None:
                tc.wndunit = unit
            else:
                if tc.wndunit == 'knot':
                    tc.wndunit = 'm/s'
                else:
                    tc.wndunit = 'knot'
        
        return self
    
    
    def __len__(self):
        """
        Get the number of TCs.
        """
        return self.TCs.__len__()
    
    def __getitem__(self, key):
        """
        Used to iterate over the whole TCSet.
        """
        return self.TCs[key]
    
    def __repr__(self):
        """
        Used to print the TC dataset.
        """
        info = []
        
        if self.agency == None:
            info.append('TC best-track dataset:\n')
        else:
            info.append('TC best-track dataset ({0:s}):\n'.format(self.agency))
            
        info.append('  {0:1d} TCs from {1:4d} to {2:4d}, {3:6.1f} cyclone days\n'
                    .format(len(self.TCs), self.TCs[0].year,
                            self.TCs[-1].year, self.total_cyclone_days()))
        
        minTC, maxTC = self.__find_duration_extrema()
        
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
        
        for tc in self.TCs:
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
    
    def __find_duration_extrema(self):
        """
        Find minimum and maximum durations of TCs.
        
        Returns
        ----------
        minTC, maxTC: tuple
            Minimum and maximum duration TCs
        """
        minDr, maxDr = -undef, undef
        minTC, maxTC =  None , None
        
        for tc in self.TCs:
            duration = tc.duration()
            
            if duration < minDr:
                minDr = duration
                minTC = tc
                
            if duration > maxDr:
                maxDr = duration
                maxTC = tc
        
        return minTC, maxTC


class TC(object):
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
        self.ID = ID
        self.name = name
        self.year = year
        self.records = records
        self.wndunit = wndunit.lower()
        self.fcstTime = fcstTime
        
        if wndunit not in ['knot', 'm/s']:
            raise Exception('invalid wind unit, should be "knot" or "m/s"')
    
    
    def sel(self, cond):
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
        
        Returns
        ----------
        re: TC
            A new TC containing the records satisfying the conditions
        """
        re = self.copy(copy_records=False)
        
        re.records = self.records.loc[cond]
        
        return re
    
    
    def duration(self):
        """
        Get duration (days) of this TC.

        Returns
        ----------
        re: float
            Duration of this TC in unit of day
        """
        duration = np.ptp(self.records['TIME'])
        return duration.astype('timedelta64[h]').astype(int) / 24.0
    
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
        
        minP, Ppos = min(prs), prs.argmin()
        maxW, Wpos = max(wnd), wnd.argmax()
        
        if minP != undef and maxW != undef:
            if Ppos != Wpos:
                if prs[Wpos] == minP:
                    return minP, maxW
                elif wnd[Ppos] == maxW:
                    return minP, maxW
                else:
                    return min(prs[wnd==maxW]), maxW
        
        return minP, maxW
    
    
    def plot_track(self, **kwargs):
        """
        Plot the track of the TC.
        """
        return plot_track(self, **kwargs)
    
    def plot_intensity(self, unit='knot', **kwargs):
        """
        Plot the intensity of the TC.
        """
        return plot_intensity(self, **kwargs)
    
    def plot(self, **kwargs):
        """
        Plot the track and intensity of the TC.
        """
        plot(self, **kwargs)
    
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
            re = TC(s.ID, s.name, s.year, s.wndunit, s.fcstTime,
                    s.records.copy())
        else:
            re = TC(s.ID, s.name, s.year, s.wndunit, s.fcstTime, None)
        
        return re
        
        
    
    def __len__(self):
        """
        Get the number of records.
        """
        return self.records.__len__()
    
    def __iter__(self):
        return self.records.itertuples()
    
    
    def __getitem__(self, key):
        """
        Used to iterate over the TC record.
        """
        if isinstance(key, int):
            return self.records.iloc[[key]]
        elif isinstance(key, str):
            return self.records[key]
        else:
            raise Exception('invalid type of key, should be int or str')
    
    def __repr__(self):
        """
        Used to print the TC.
        """
        s = self
        
        info = 'TC (ID={0:s}, name={1:s}, year={2:4d}, fcstTime={3:02d})\n' \
                .format(s.ID, s.name, s.year, s.fcstTime)
        
        return info + s.records.__repr__()


"""
Helper (private) methods are defined below
"""


"""
Test codes
"""
if __name__ == '__main__':
    print('ok')

