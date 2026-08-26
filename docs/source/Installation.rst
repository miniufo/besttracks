Installation
============

Requirements
^^^^^^^^^^^^

besttracks is compatible with python 3 (>= version 3.8). It requires
numpy_, pandas_, matplotlib_, cartopy_, xarray_, and scikit-learn_.

Installation from pip
^^^^^^^^^^^^^^^^^^^^^

One can do this by using pip::

    pip install besttracks

This will install the latest release from
`pypi <https://pypi.python.org/pypi>`_.

Installation from github
^^^^^^^^^^^^^^^^^^^^^^^^

besttracks is still under active development. To obtain the latest development
version, you may clone the `source repository
<https://github.com/miniufo/besttracks>`_ and install it::

    git clone https://github.com/miniufo/besttracks.git
    cd besttracks
    python setup.py install

or simply::

    pip install git+https://github.com/miniufo/besttracks.git


How to run the notebooks
^^^^^^^^^^^^^^^^^^^^^^^^

If you want to run the example notebooks in this documentation, you will need
a few extra dependencies that you can install via::

    conda env create -f environment.yml
    conda activate besttracks


.. _numpy: https://numpy.org/
.. _pandas: https://pandas.pydata.org/
.. _matplotlib: https://matplotlib.org/
.. _cartopy: https://scitools.org.uk/cartopy/docs/latest/
.. _xarray: http://xarray.pydata.org/en/stable/
.. _scikit-learn: https://scikit-learn.org/
