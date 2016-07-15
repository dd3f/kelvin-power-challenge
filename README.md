# kelvin-power-challenge

A baseline solution and visualization tools for [Kelvin Mars Power Challenge](https://kelvins.esa.int/mars-express-power-challenge/).

### Instructions

#### Prerequisites
- Python 2.7
- Numpy
- Pandas (I am using 0.18, if you cannot upgrade you need to change .resample() methods to old syntax)
- Matplotlib

A convenient way to get all prerequisites is to install the [Anaconda](https://www.continuum.io/downloads) Python distribution.

#### Download source and competition data

1. Clone the repository using git: `git clone https://github.com/alex-bauer/kelvin-power-challenge.git`
2. Download data file from [competition website](https://kelvins.esa.int/mars-express-power-challenge/data/) and unpack to data folder

#### Generate baseline solutions

3. Run `python mean_baseline.py`
4. Run `python rf_baseline.py` (0.12 on public leaderboard)

