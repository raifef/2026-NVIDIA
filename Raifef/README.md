Finalphaseoutliers.ipynb is to perform the QE-MTS for DCQO and DAQO (gpu or cpu enabled, choose the backend). It is efficient to perform statistically many runs (>100) for N >~ 24 in a reasonable timeframe, from which the outliers can be deduced. this is also done in this file.
The speed comparison is in speedtest.ipynb
Plotting is performed with plotrecovery.py
Test suite is test_phase_ab_sanity.py
