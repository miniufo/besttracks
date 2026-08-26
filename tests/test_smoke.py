# -*- coding: utf-8 -*-
"""
Smoke tests for besttracks: verify package import and basic Particle/TC/TCSet
functionality using synthetic data.

These tests do NOT require local Data/ files or hardcoded paths — they
are designed to run on GitHub CI with only pip-installed dependencies.
"""
import numpy as np
import pandas as pd
import pytest

from besttracks import Particle, TC, Drifter, ParticleSet, TCSet, DrifterSet


# ---------------------------------------------------------------------------
# Fixtures: synthetic TC trajectory records
# ---------------------------------------------------------------------------

def _make_tc_records(n=5, start_lon=125.0, start_lat=18.0):
    """Create a small DataFrame mimicking a TC best-track record."""
    times = pd.date_range('2024-09-01', periods=n, freq='6h')
    lons = start_lon + np.arange(n) * 0.5
    lats = start_lat + np.arange(n) * 0.3
    wnd = np.array([20, 30, 40, 35, 25], dtype=float)
    prs = np.array([1000, 990, 980, 985, 995], dtype=float)

    return pd.DataFrame({
        'TIME': times,
        'LON': lons,
        'LAT': lats,
        'WND': wnd,
        'PRS': prs,
    })


@pytest.fixture
def tc_records():
    return _make_tc_records()


@pytest.fixture
def single_tc(tc_records):
    """A single TC object with synthetic records."""
    return TC(ID='2024001', name='TESTTC', year='2024',
              wndunit='knot', fcstTime=0, records=tc_records)


@pytest.fixture
def single_particle(tc_records):
    """A generic Particle with the same records."""
    return Particle(ID='drifter001', records=tc_records)


# ---------------------------------------------------------------------------
# Import tests
# ---------------------------------------------------------------------------

class TestImport:
    def test_import_particle(self):
        assert Particle is not None

    def test_import_tc(self):
        assert TC is not None

    def test_import_drifter(self):
        assert Drifter is not None

    def test_import_sets(self):
        assert TCSet is not None
        assert DrifterSet is not None


# ---------------------------------------------------------------------------
# Particle tests
# ---------------------------------------------------------------------------

class TestParticle:
    def test_init(self, single_particle):
        p = single_particle
        assert p.ID == 'drifter001'
        assert len(p.records) == 5

    def test_len(self, single_particle):
        assert len(single_particle) == 5

    def test_sel(self, single_particle):
        """sel() filters records by a condition."""
        p = single_particle
        sub = p.sel(lambda df: df['WND'] > 25)
        # WND=[20,30,40,35,25], >25 → 30,40,35 = 3 records
        assert len(sub) == 3

    def test_copy(self, single_particle):
        p2 = single_particle.copy()
        assert p2.ID == single_particle.ID
        assert len(p2.records) == len(single_particle.records)


# ---------------------------------------------------------------------------
# TC tests
# ---------------------------------------------------------------------------

class TestTC:
    def test_init(self, single_tc):
        tc = single_tc
        assert tc.ID == '2024001'
        assert tc.name == 'TESTTC'
        assert tc.year == '2024'
        assert tc.wndunit == 'knot'
        assert tc.fcstTime == 0

    def test_invalid_wind_unit(self, tc_records):
        with pytest.raises(Exception):
            TC(ID='2024001', name='TESTTC', year='2024',
               wndunit='mph', fcstTime=0, records=tc_records)

    def test_duration(self, single_tc):
        """duration returns the time span of the TC in days."""
        dur = single_tc.duration()
        assert dur is not None
        # 5 records at 6h intervals → 24 hours = 1.0 day
        assert abs(float(dur) - 1.0) < 1e-6

    def test_peak_intensity(self, single_tc):
        """peak_intensity returns max wind."""
        peak = single_tc.peak_intensity()
        assert peak is not None

    def test_change_wind_unit(self, single_tc):
        """Wind unit can be converted from knot to m/s."""
        original_wnd = single_tc.records['WND'].values.copy()
        single_tc.change_wind_unit(unit='m/s')
        assert single_tc.wndunit == 'm/s'
        # knot → m/s: 1 knot = 0.51444 m/s, so values should decrease
        converted = single_tc.records['WND'].values
        # skip undef entries; all fixture winds are valid
        valid = original_wnd != -9999.0
        assert np.all(converted[valid] < original_wnd[valid])


# ---------------------------------------------------------------------------
# ParticleSet / TCSet tests
# ---------------------------------------------------------------------------

class TestTCSet:
    def test_init(self, single_tc):
        """TCSet can be constructed from a list of TCs."""
        tc2 = single_tc.copy()
        tc2.ID = '2024002'
        tcs = TCSet([single_tc, tc2], agency='JTWC')
        assert len(tcs) == 2

    def test_groupby(self, single_tc):
        """groupby groups TCs by a field."""
        tc2 = single_tc.copy()
        tc2.ID = '2024002'
        tc2.name = 'SECOND'
        tcs = TCSet([single_tc, tc2], agency='JTWC')
        grouped = tcs.groupby('year')
        assert grouped is not None
