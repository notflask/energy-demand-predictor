# Energy Demand Predictor

A personal project where I'm learning how to forecast electricity demand using Python and machine learning.

The idea is to use historical electricity consumption data from Germany to predict future demand. I'm starting with basic statistics and simple forecasting methods before moving on to machine learning.

## Current progress

So far, I've implemented:

- Basic statistics, normalization and moving averages with NumPy
- A simple forecasting baseline that predicts the next value using the previous one
- Exploratory data analysis with Pandas
- Hourly and weekly electricity demand visualizations with Matplotlib
- Unit tests using pytest

I'm currently working with historical German electricity demand data from 2015 to 2020.

## Tech stack

- Python
- NumPy
- Pandas
- Matplotlib
- pytest
- uv

I'll be adding scikit-learn once I start training models.

## Getting started

Clone the repository:

```bash
git clone https://github.com/notflask/energy-demand-predictor.git
cd energy-demand-predictor
```

Install dependencies:

```bash
uv sync
```

Run tests:

```bash
uv run pytest
```

### Dataset

The data comes from [Open Power System Data](https://data.open-power-system-data.org/time_series/2020-10-06/).

Download `time_series_60min_singleindex.csv` and place it in `data/raw/`. The CSV isn't included in the repository because of its size.

## What's next?

- Create time-series features using historical demand
- Train linear regression and Random Forest models
- Compare their predictions against simple baselines
- Evaluate performance using MAE and RMSE
- Make sure the evaluation doesn't suffer from data leakage

## Why I'm building this

I wanted to work on something that combines programming, mathematics and real-world data. Electricity demand forecasting seemed like a good problem to explore because it involves patterns, statistics and predictions that can actually be useful.

I'm building this project from scratch to understand how machine learning works, rather than just training a model and looking at its accuracy.