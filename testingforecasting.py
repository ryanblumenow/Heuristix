import streamlit as st

# Data manipulation
# ==============================================================================
import numpy as np
import pandas as pd

# Plots
# ==============================================================================
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5

# Modeling and Forecasting
# ==============================================================================
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregMultiOutput import ForecasterAutoregMultiOutput
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster

from joblib import dump, load

exog_var = st.selectbox("Do you want an exogenous variable?", options=["Yes", "No"])

# Warnings configuration
# ==============================================================================
import warnings
# warnings.filterwarnings('ignore')

# Data ingestion
# ==============================================================================
url = 'https://raw.githubusercontent.com/JoaquinAmatRodrigo/skforecast/master/data/h2o_exog.csv'
data = pd.read_csv(url, sep=',')

# data['LAST_INTERACTION'] = pd.to_datetime(data['LAST_INTERACTION'], format='%d/%m/%Y')

# data.set_index("LAST_INTERACTION").groupby([pd.Grouper(freq="D"), "SALES"]).sum().reset_index()

# data['AGE'].fillna(0)

# st.write(data.LAST_INTERACTION.count())
# st.write(data.SALES.count())
# st.write(data.AGE.count())

st.write(data.head())

# data = data.reset_index()

# Data preparation
# ==============================================================================
data = data.rename(columns={'fecha': 'date'})
# data = data.rename(columns={'LAST_INTERACTION': 'date'})
data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')
data = data.set_index('date')
data = data.rename(columns={'x': 'y'})
# data = data.rename(columns={'SALES': 'y'})
# data = data.rename(columns={'AGE': 'exog_1'})
data = data.asfreq('MS')
data = data.sort_index()
st.write(data.head())

st.write(f'Number of rows with missing values: {data.isnull().any(axis=1).mean()}')

# Verify that a temporary index is complete
# ==============================================================================
(data.index == pd.date_range(start=data.index.min(),
                             end=data.index.max(),
                             freq=data.index.freq)).all()

# Fill gaps in a temporary index
# ==============================================================================
data.asfreq(freq='D', fill_value=np.nan)

# Split data into train-test
# ==============================================================================
steps = 36
data_train = data[:-steps]
data_test  = data[-steps:]

print(f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
print(f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

fig, ax=plt.subplots(figsize=(9, 4))
data_train['y'].plot(ax=ax, label='train')
data_test['y'].plot(ax=ax, label='test')

if exog_var == "Yes":

    data_train['exog_1'].plot(ax=ax, label='train_exog')
    data_test['exog_1'].plot(ax=ax, label='test_exog')

ax.legend()

st.pyplot(fig)

# 1. Create and train forecaster with 8 lags
# ==============================================================================
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags = 8
                )

# Where no exogenous data for future periods is available

if exog_var == "No":

    forecaster.fit(y=data_train['y'])

# Where exogenous data for future periods is available

if exog_var == "Yes":

    forecaster.fit(y=data_train['y'], exog=data_train['exog_1'])

# Predictions
# ==============================================================================
steps = 36
if exog_var == "No":
    predictions = forecaster.predict(steps=steps) # no future data for exog
if exog_var == "Yes":
    predictions = forecaster.predict(steps=steps, exog=data_test['exog_1']) # future data for exog

st.table(predictions.tail(5))

# Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
data_train['y'].plot(ax=ax, label='train')
data_test['y'].plot(ax=ax, label='test')
predictions.plot(ax=ax, label='predictions')
ax.legend()

st.pyplot(fig)

# Test error
# ==============================================================================
error_mse = mean_squared_error(
                y_true = data_test['y'],
                y_pred = predictions
            )

st.write(f"Test error (mse): {error_mse}")

# Hyperparameter grid search to identify the best combination of lags and hyperparameters
# ==============================================================================
steps = 36
forecaster = ForecasterAutoreg(
                regressor = RandomForestRegressor(random_state=123),
                lags      = 12 # This value will be replaced in the grid search
             )

# Lags used as predictors
lags_grid = [10, 20]

# Regressor's hyperparameters
param_grid = {'n_estimators': [100, 500],
              'max_depth': [3, 5, 10]}

if exog_var == "No":

    results_grid = grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = data_train['y'],
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        steps              = steps,
                        refit              = True,
                        metric             = 'mean_squared_error',
                        initial_train_size = int(len(data_train)*0.5),
                        fixed_train_size   = False,
                        return_best        = True,
                        verbose            = False
               )

if exog_var == "Yes":

    results_grid = grid_search_forecaster(
                        forecaster  = forecaster,
                        y           = data_train['y'],
                        exog        = data_train['exog_1'],
                        param_grid  = param_grid,
                        lags_grid   = lags_grid,
                        steps       = steps,
                        refit       = True,
                        metric      = 'mean_squared_error',
                        initial_train_size = int(len(data_train)*0.5),
                        return_best = True,
                        verbose     = False
               )

# Grid Search results
# ==============================================================================
st.write(results_grid)

# Create and train forecaster with the best hyperparameters (not necessary if return_best=True specified above, but ensures certainty)
# ==============================================================================
# regressor = RandomForestRegressor(max_depth=3, n_estimators=500, random_state=123)
# forecaster = ForecasterAutoreg(
#                 regressor = regressor,
#                 lags      = 20
#              )

# forecaster.fit(y=data_train['y'])

# Predictions
# ==============================================================================
if exog_var =="No":

    predictions = forecaster.predict(steps=steps)

if exog_var == "Yes":

    predictions = forecaster.predict(steps=steps, exog=data_test['exog_1'])

# Plot
# ==============================================================================
fig, ax = plt.subplots(figsize=(9, 4))
data_train['y'].plot(ax=ax, label='train')
data_test['y'].plot(ax=ax, label='test')
predictions.plot(ax=ax, label='predictions')
ax.legend()

st.pyplot(fig)

# Test error
# ==============================================================================
error_mse = mean_squared_error(
                y_true = data_test['y'],
                y_pred = predictions
                )

st.write(f"Test error (mse): {error_mse}")

# Backtesting
# ==============================================================================
steps = 36
n_backtesting = 36*3 # The last 9 years are separated for the backtest

metric, predictions_backtest = backtesting_forecaster(
                                    forecaster = forecaster,
                                    y          = data['y'],
                                    initial_train_size = len(data) - n_backtesting,
                                    fixed_train_size   = False,
                                    steps      = steps,
                                    metric     = 'mean_squared_error',
                                    refit      = True,
                                    verbose    = True
                                    )

st.write(f"Backtest error: {metric}")

fig, ax = plt.subplots(figsize=(9, 4))
data.loc[predictions_backtest.index, 'y'].plot(ax=ax, label='test')
predictions_backtest.plot(ax=ax, label='predictions')
ax.legend()

st.pyplot(fig)

# Predictors importance
# ==============================================================================
st.table(forecaster.get_feature_importance())

