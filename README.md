# Predicting Stock Alpha from Fundamentals: Statistical vs ML Approaches

## Overview
This project investigates whether fundamental indicators (e.g., ROE, ROA, leverage, margins) can predict alpha in technology stocks.  
I built an end-to-end pipeline that collects financial data, engineers features, and compares statistical and machine learning models through portfolio backtesting.

## Motivation
- Recruiter-facing demonstration of data science applied in finance.  
- Showcases skills in data collection, preprocessing, modeling, and evaluation.  
- Combines academic finance methods (OLS, Fama–MacBeth) with modern ML (Linear Regression, Gradient Boosting).  

## Data
- **Universe**: 20 leading S&P 500 technology stocks (2010–2023).  
- **Fundamentals**: Net income, equity, assets, liabilities, revenue, margins (processed to ROE, ROA, leverage, profit margin, gross margin).  
- **Prices**: Quarterly adjusted stock returns.  
- **Source**: Public filings (Kaggle fundamentals dataset) + Yahoo Finance (via `yfinance`).  

## Pipeline
1. **Data Collection**  
   - S&P 500 tickers scraped.  
   - Fundamentals merged with price data.  
   - Cached locally (Colab + Google Drive).  

2. **Data Cleaning & Preprocessing**  
   - Forward-fill and NaN handling.  
   - Ratio engineering (ROE, ROA, etc.).  
   - Saved ML-ready dataset (`tech20_ml_ready.csv`).  

3. **Modeling**  
   - **OLS / Fama–MacBeth**: Statistical baseline, interpretable betas.  
   - **Linear Regression**: ML baseline with expanding-window backtest.  
   - **Gradient Boosting Regressor**: Nonlinear ML, tuned hyperparameters, expanding-window backtest.  

4. **Evaluation**  
   - Long–short portfolio returns (top 5 vs bottom 5 predicted stocks).  
   - Cumulative return curves.  
   - Summary statistics: average quarterly return, volatility, and Sharpe ratio.  

## Results
- **OLS / Fama–MacBeth**: No significant predictors; weak explanatory power.  
- **Linear Regression**: Negative cumulative return (≈ –15%).  
- **Gradient Boosting**: Positive cumulative return (≈ +8%), Sharpe ratio ≈ 0.2.  
- **Takeaway**: Fundamentals alone do not robustly predict alpha, but ML can extract weak signals.  


| Model              | Final Cumulative Return | Avg Quarterly Return | Volatility | Sharpe Ratio |
|--------------------|--------------------------|-----------------------|-------------|---------------|
| OLS                | 0.855                    | –0.016                | 0.085       | –0.19         |
| Linear Regression  | 0.855                    | –0.016                | 0.085       | –0.19         |
| Gradient Boosting  | 1.086                    | 0.011                 | 0.051       | 0.22          |

## Tech Stack
- **Python**: pandas, numpy, matplotlib  
- **Finance data**: yfinance, fundamentals CSV (Kaggle)  
- **Modeling**: statsmodels, scikit-learn (LinearRegression, GradientBoostingRegressor)  

## Key Learnings
- End-to-end data science pipeline in finance.  
- Importance of preprocessing (NaNs, ratios).  
- Backtesting as evaluation, not just RMSE.  
- ML (Gradient Boosting) can outperform linear methods, but signals remain weak.  

## Next Steps
- Extend to all S&P 500 stocks.  
- Try regularized models (Lasso, Ridge).  
- Add benchmarks (e.g., XLK ETF excess returns).  
- Test alternative horizons (monthly vs quarterly).  

## Author
Karthik Velavarthypathi
