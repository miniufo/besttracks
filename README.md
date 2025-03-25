# besttracks

![tracks plot](https://raw.githubusercontent.com/miniufo/besttracks/master/pics/Global_TC_tracks.png)


## 1. Introduction
Tropical cyclone (TC) best-track datasets are analyzed, maintained, and hosted by several Regional Specialized Meteorological Centers (RSMCs), agencies, or projects all over the world.  These agencies include:
-  **JTWC:** Joint Typhoon Warning Center, Naval Oceanography Portal.  This agency currently hosted TC datasets over several ocean basins except the North Atlantic Ocean i.e.,  western Pacific basin (BWP), North Indian Ocean (BIO), and Southern Hemisphere basin (BSH).
https://www.metoc.navy.mil/jtwc/jtwc.html?best-tracks
-  **CMA:** China Meteorological Administration.  This agency only hosted the TC dataset over the western North Pacific.
http://tcdata.typhoon.org.cn/en/zjljsjj_zlhq.html
- **JMA:** RSMC Tokyo-Typhoon Center, Japan Meteorological Agency.  This agency only hosted the TC dataset over the western North Pacific
https://www.jma.go.jp/jma/jma-eng/jma-center/rsmc-hp-pub-eg/trackarchives.html
- **NHC:** National Hurricane Center, National Oceanic and Atmospheric Administration.  This agency hosted the TC datasets for both the North Atlantic Ocean and eastern North Pacific, which are not covered by JTWC.
https://www.nhc.noaa.gov/data/#hurdat
- **IBTrACS:** International Best Track Archive for Climate Stewardship.  This project merges the best-track datasets already exist at other agencies (more than the above) into a worldwide TC database.
https://www.ncdc.noaa.gov/ibtracs/


| RSMC | WNP | NEP | NAT | NIO | SIO | WSP | SAT |
| :----: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| JTWC | X |  |  | X | X | X |  |
| CMA | X |  |  |  |  |  |  |
| JMA | X |  |  |  |  |  |  |
| NHC |  | X | X |  |  |  |  |
| IBTrACS | X | X | X | X | X |  | X |


Unfortunately, different agencies use different data formats.  This python-based project **`besttracks`**, aims to provide a unified interface to access these datasets in different formats, and organizes them into a unified data structure called **`TCSet`** and **`TC`**, which are based on `pandas.DataFrame` that are convient for python users.  Simple plot of track and intensity is also easy and some basic statistics are also provided.

Up to now, the datasets from the above agencies are supported.  It would be good to add more agencies and more formats.  We also provide the parser function for CMA operational forecast data (BABJ format), which is also popular in China.

---

## 2. How to install
**Requirements**
`besttracks` is developed under the environment with `numpy` (=version 1.15.4), `pandas` (=version 1.0.3), `xarray` (=version 0.15.1), `matplotlib` (=version 3.3.1), and `cartopy` (=version 0.18.0).  Older versions of these packages are not well tested.


**Install from github**
```
git clone https://github.com/miniufo/besttracks.git
```
---

## 3. Data Structures
The core data structures of `besttracks` are **`TCSet`** and **`TC`**:

### 3.1 `TCSet` (Tropical Cyclone List)
A container for a list of tropical cyclones (`TC`). Each `TCSet` object represents a collection of tropical cyclones parsed from a specific dataset.

#### Structure:
```plaintext
TCSet
└── TC Object (Example)
    ├── ID: 1979002S04179 (str)
    ├── ace() (Method)
    │   └── Signature: ()
    ├── binning() (Method)
    │   └── Signature: (var=None, **kwargs)
    ├── change_wind_unit() (Method)
    │   └── Signature: (unit=None)
    ├── copy() (Method)
    │   └── Signature: (copy_records=True)
    ├── duration() (Method)
    │   └── Signature: ()
    ├── fcstTime: 0 (int)
    ├── get_as_xarray() (Method)
    │   └── Signature: (field)
    ├── name: GORDON (str)
    ├── peak_intensity() (Method)
    │   └── Signature: ()
    ├── plot() (Method)
    │   └── Signature: (**kwargs)
    ├── plot_intensity() (Method)
    │   └── Signature: (unit='knot', **kwargs)
    ├── plot_track() (Method)
    │   └── Signature: (**kwargs)
    ├── records (DataFrame)
    │   └── Shape: (101, 6), Columns: ['TIME', 'TYPE', 'LAT', 'LON', 'WND', 'PRS']
    ├── resample() (Method)
    │   └── Signature: (*args, **kwargs)
    ├── sel() (Method)
    │   └── Signature: (cond, copy=True)
    ├── translate_velocity() (Method)
    │   └── Signature: (Rearth=6371200)
    ├── wndunit: knot (str)
    └── year: 1979 (int)
```
### 3.2 TC (Single Tropical Cyclone)
A single tropical cyclone object that contains detailed information about the cyclone's track, intensity, and metadata.

#### Key attributes and methods:

- **ID**: Unique identifier for the tropical cyclone.
- **name**: Name of the tropical cyclone.
- **records**: A `pandas.DataFrame` containing the time-series data of the cyclone's track and intensity.
  - Columns: `['TIME', 'TYPE', 'LAT', 'LON', 'WND', 'PRS']`
- **Methods**:
  - `plot()`: Plot the track and intensity of the cyclone.
  - `binning()`: Bin the track data into gridded statistics.
  - `ace()`: Calculate the Accumulated Cyclone Energy (ACE).
  - `duration()`: Get the total duration of the cyclone.
  - `peak_intensity()`: Find the peak intensity of the cyclone.
  - `change_wind_unit(unit)`: Convert wind speed units between knots, meters per second, and kilometers per hour.
---

## 4. Examples
### 4.1 Best-track datasets manipulations
Parsing best-track dataset **CMA** into `TCSet` would be as simple as:
```python
from besttracks import parse_TCs

# parse dataset from CMA
TCs_CMA = parse_TCs('./CH*.txt', agency='CMA')

# Brief describe the dataset
print(TCs_CMA)

# Plotting all TC tracks
TCs_CMA.plot_tracks()
```

![tracks plot](https://raw.githubusercontent.com/miniufo/besttracks/master/pics/tracks_cma.png)

One can also bin the tracks into gridded statistics (also known as PDF distribution) as:
```python
# binning the tracks into gridded data
TCs_CMA.binning()
```

![binning plot](https://raw.githubusercontent.com/miniufo/besttracks/master/pics/binning_cma.png)

---

### 4.2 A single TC manipulation
Manipulating a single `TC` is also simple:
```python
# Selecting a single TC
tc = TCs_cma[-1]

# Briefly descibe the TC
print(tc)

# Plot the TC track and intensity
tc.plot()
```
![tc plot](https://raw.githubusercontent.com/miniufo/besttracks/master/pics/tc_plot.png)

---

### 4.3 Timeseries statistics
`TCSet` also supports statistical analysis over time space. One can plot the timeseries of TC number and accumulated cyclonic energy (ACE) of a `TCSet` as:
```python
# plot the climatological timeseries of No. and ACE
TCs_CMA.plot_timeseries(freq='annual')
```
![tc plot](https://raw.githubusercontent.com/miniufo/besttracks/master/pics/timeseries.png)


More examples can be found at this [notebook](https://github.com/miniufo/besttracks/blob/master/notebooks/QuickGuide.ipynb)
