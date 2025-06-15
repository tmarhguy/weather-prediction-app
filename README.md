Weather Prediction App — Next-Day Temperature Forecast

This project predicts the next day’s maximum temperature in Oakland, California using historical weather data from 1948 to 2025. It applies regression techniques with cleaned meteorological features and engineered temporal indicators.

---

Project Structure

```
weather-prediction-app/
├── data/
│   └── local_weather_oakland.csv      # Historical weather data
├── WeatherPrediction.ipynb            # Main notebook (data prep + modeling)
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

Problem Statement

Accurately forecasting next-day temperature has significant implications for:

- Agriculture 🌾
- City planning 🏙️
- Daily lifestyle 🌡️

Using classical machine learning, we aim to train a regression model on cleaned weather records to minimize prediction error and improve planning decisions.

---

Data Source & Cleaning

- Source: NOAA’s Oakland weather station (USW00023230)
- Date Range: Jan 1, 1948 → Jun 10, 2025
- Original columns: 37
- Selected features:
  - `PRCP`: Precipitation
  - `TMAX`: Max temperature
  - `TMIN`: Min temperature
  - `SNOW`, `SNWD`: Later dropped due to sparseness

Cleaning Steps

- Missing `precipitation` filled with `0.0`
- Forward fill for `TMAX` and `TMIN`
- Shifted temperature used to predict next day's `TMAX`
- Added time-based features for context:
  - `month`
  - `dayofyear`
  - `prev_precip` (lagged feature)

---

Feature Engineering

| Feature         | Description                    |
| --------------- | ------------------------------ |
| `max_temp`      | Max temp of current day (°C)   |
| `min_temp`      | Min temp of current day (°C)   |
| `precipitation` | Current day precipitation (mm) |
| `prev_precip`   | Previous day's precipitation   |
| `month`         | Extracted from datetime index  |
| `dayofyear`     | Helps model seasonality        |

---

Modeling Approach

- Model: Ridge Regression
- Library: `scikit-learn`
- Train set: Until Dec 31, 2020
- Test set: Jan 1, 2021 – Jun 10, 2025

```python
from sklearn.linear_model import Ridge
reg = Ridge(alpha=0.1)
reg.fit(train[predictors], train["target"])
predictions = reg.predict(test[predictors])
```

---

Performance Metrics

| Metric                         | Value   |
| ------------------------------ | ------- |
| MAE (Mean Absolute Error)      | 1.85 °C |
| RMSE (Root Mean Squared Error) | 2.53 °C |

> The model performs well for classical regression using limited numerical features. Errors below 2 °C are considered good for next-day forecasts in real-world deployments.

---

Sample Predictions

| Date       | Actual Temp (°C) | Predicted Temp (°C) |
| ---------- | ---------------- | ------------------- |
| 2021-01-01 | 13.9             | 15.39               |
| 2021-01-02 | 13.3             | 15.20               |
| 2021-01-03 | 16.7             | 14.60               |
| ...        | ...              | ...                 |
| 2025-06-10 | 16.7             | 18.10               |

Dependencies

```text
pandas
numpy
scikit-learn
matplotlib
```

Install them using:

```bash
pip install -r requirements.txt
```

---

Future Improvements

- Use models like Random Forest or XGBoost
- Add rolling averages and weather condition flags
- Explore classification: “Will tomorrow be hotter than today?”
- Visualize confidence intervals and seasonal trends

---

Author

Tyrone Marhguy  
Computer Engineering @ University of Pennsylvania  
🔗 [GitHub](https://github.com/tmarhguy/)
