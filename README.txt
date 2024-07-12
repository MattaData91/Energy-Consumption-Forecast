 Time Series Forecasting with Machine Learning

 Using Machine Learning to Forecast Energy Consumption

This notebook demonstrates how to forecast energy consumption using machine learning techniques. The data used in this analysis is obtained from the [Hourly Energy Consumption dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?resource=download) on Kaggle.

 # 1. Imports and Data Loading


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error

color_pal = sns.color_palette()
plt.style.use('fivethirtyeight')

df = pd.read_csv('../input/hourly-energy-consumption/PJME_hourly.csv')
df = df.set_index('Datetime')
df.index = pd.to_datetime(df.index)


## *Imports libraries*: Necessary libraries for data manipulation, visualization, and machine learning.
## *Loads dataset*: Reads the dataset and sets the index to 'Datetime' for time series analysis.

# 2. Plotting Energy Use


df.plot(style='.', figsize=(15, 5), color=color_pal[0], title='PJME Energy Use in MW')
plt.show()


## *Visualization*: Plots the entire time series data to observe trends and patterns.

#  3. Train / Test Split


train = df.loc[df.index < '01-01-2015']
test = df.loc[df.index >= '01-01-2015']

fig, ax = plt.subplots(figsize=(15, 5))
train.plot(ax=ax, label='Training Set', title='Data Train/Test Split')
test.plot(ax=ax, label='Test Set')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()


## *Splits data*: Divides the data into training and test sets based on a specific date.
## *Plots split data*: Visualizes the split to ensure the correct separation of training and testing data.

#  4. Weekly Data Plot


df.loc[(df.index > '01-01-2010') & (df.index < '01-08-2010')].plot(figsize=(15, 5), title='Week Of Data')
plt.show()


## *Weekly data visualization*: Plots a subset of the data to analyze patterns over a week.

#  5. Feature Creation


def create_features(df):
    df = df.copy()
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['quarter'] = df.index.quarter
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['dayofmonth'] = df.index.day
    df['weekofyear'] = df.index.isocalendar().week
    return df

df = create_features(df)


## *Feature engineering*: Creates new time-based features (hour, day, month, etc.) from the datetime index to use in the model.

#  6. Feature/Target Relationship Visualization


fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='hour', y='PJME_MW')
ax.set_title('MW by Hour')
plt.show()

fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(data=df, x='month', y='PJME_MW', palette='Blues')
ax.set_title('MW by Month')
plt.show()


## *Visualizes feature relationships*: Uses box plots to show how energy consumption varies by hour and month.

#  7. Model Creation


train = create_features(train)
test = create_features(test)

FEATURES = ['dayofyear', 'hour', 'dayofweek', 'quarter', 'month', 'year']
TARGET = 'PJME_MW'

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

reg = xgb.XGBRegressor(
    base_score=0.5, booster='gbtree', n_estimators=1000,
    early_stopping_rounds=50, objective='reg:linear', max_depth=3,
    learning_rate=0.01
)
reg.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=100)


## *Model training*: Trains an XGBoost regressor on the training data, using engineered features.

#  8. Feature Importance


fi = pd.DataFrame(data=reg.feature_importances_, index=reg.feature_names_in_, columns=['importance'])
fi.sort_values('importance').plot(kind='barh', title='Feature Importance')
plt.show()


## *Feature importance*: Plots the importance of each feature in the model to understand their impact on predictions.

#  9. Forecast on Test Set


test['prediction'] = reg.predict(X_test)
df = df.merge(test[['prediction']], how='left', left_index=True, right_index=True)

ax = df[['PJME_MW']].plot(figsize=(15, 5))
df['prediction'].plot(ax=ax, style='.')
plt.legend(['Truth Data', 'Predictions'])
ax.set_title('Raw Data and Prediction')
plt.show()

ax = df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['PJME_MW'].plot(figsize=(15, 5), title='Week Of Data')
df.loc[(df.index > '04-01-2018') & (df.index < '04-08-2018')]['prediction'].plot(style='.')
plt.legend(['Truth Data','Prediction'])
plt.show()


## *Prediction visualization*: Compares the predicted values to the actual values to evaluate model performance.

#  10. Score (RMSE)


score = np.sqrt(mean_squared_error(test['PJME_MW'], test['prediction']))
print(f'RMSE Score on Test set: {score:0.2f}')


## *Model evaluation*: Calculates the root mean squared error (RMSE) to quantify the model's accuracy.

#  11. Calculate Error


test['error'] = np.abs(test[TARGET] - test['prediction'])
test['date'] = test.index.date
test.groupby(['date'])['error'].mean().sort_values(ascending=False).head(10)


## *Error analysis*: Identifies the dates with the highest prediction errors to understand model limitations.

## Next Steps
- More robust cross-validation
- Add more features (weather forecast, holidays)

### For further details, visit the [Hourly Energy Consumption dataset](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption?resource=download).