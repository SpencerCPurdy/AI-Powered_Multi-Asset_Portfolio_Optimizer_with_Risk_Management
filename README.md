# AI-Powered Multi-Asset Portfolio Optimizer with Risk Management

A production-grade portfolio optimization system that combines classical financial theory with modern machine learning. The system integrates Modern Portfolio Theory, the Black-Litterman model, XGBoost and LSTM return forecasting, Fama-French multi-factor analysis, FinBERT sentiment analysis, and advanced risk metrics into a unified pipeline for multi-asset portfolio construction and management.

## About

This portfolio project demonstrates quantitative finance engineering by implementing an institutional-grade optimization framework across a 10-asset ETF universe. Pre-trained ML models generate return forecasts that feed directly into the optimizer, with all signals — historical, factor-based, ML-predicted, and sentiment-driven — blended into actionable allocations via convex optimization.

**Author:** Spencer Purdy  
**Development Environment:** Google Colab Pro (H100 GPU, High RAM)

## Features

- **Modern Portfolio Theory (MPT):** Mean-variance optimization with configurable risk aversion, efficient frontier computation, and min/max weight constraints via CVXPY
- **Black-Litterman Model:** Bayesian posterior estimation combining market equilibrium returns with investor views; configurable view confidence (Omega matrix)
- **ML Alpha Generation:** Pre-trained XGBoost and LSTM models for return prediction; blended 70% historical / 30% ML signal at optimization time
- **Fama-French 5-Factor Model:** OLS regression factor loadings for each asset across Market, SMB, HML, RMW, and CMA factors with systematic vs. idiosyncratic risk decomposition
- **FinBERT Sentiment Analysis:** Transformer-based financial sentiment scoring on Finnhub news over a 30-day rolling window; scores integrated as portfolio view adjustments
- **Advanced Risk Metrics:** Historical and parametric VaR/CVaR (95% confidence), Sharpe ratio, Sortino ratio, Information ratio, and maximum drawdown
- **Monte Carlo Simulation:** Up to 50,000 correlated return paths via Cholesky decomposition; outputs terminal wealth distributions, probability of loss, and percentile bands
- **Transaction Cost Optimization:** Turnover-aware rebalancing with 5% drift threshold and 10 bps per trade cost model
- **Interactive Interface:** Gradio web application with seven tabs covering optimization, risk analysis, factor exposure, Monte Carlo, performance attribution, sentiment, and documentation

## Asset Universe

| Ticker | Asset Class |
|--------|-------------|
| SPY | US Large-Cap Equity |
| QQQ | US Technology Equity |
| IWM | US Small-Cap Equity |
| EFA | International Developed Equity |
| EEM | Emerging Market Equity |
| TLT | Long-Term US Treasuries |
| IEF | Intermediate US Treasuries |
| GLD | Gold |
| SLV | Silver |
| VNQ | US Real Estate (REIT) |

**Historical Data:** 5 years | 1,253 trading days

## Model Performance

### XGBoost Return Predictor

| Metric | Train | Test |
|--------|-------|------|
| MSE | 9.57e-06 | 2.27e-04 |
| MAE | 0.0024 | 0.0128 |
| R² | 0.874 | -3.130 |
| Directional Accuracy | — | 38.4% |

**Features:** 130 engineered features (rolling returns, volatility, moving averages, RSI, volume momentum across all 10 assets)  
**Training Samples:** 843 | **Test Samples:** 211

The negative test R² and below-chance directional accuracy reflect the well-documented difficulty of cross-sectional ETF return prediction. The model functions as a weak signal blended with historical estimates rather than as a standalone predictor.

**Top 10 Features by Importance:**

| Feature | Importance |
|---------|-----------|
| SPY_ma_200d | 0.0295 |
| SPY_volatility_20d | 0.0274 |
| TLT_ma_50d | 0.0252 |
| EFA_ma_50d | 0.0237 |
| VNQ_return_60d | 0.0216 |
| TLT_price_to_ma20 | 0.0197 |
| VNQ_price_to_ma20 | 0.0194 |
| EFA_rsi | 0.0192 |
| IWM_ma_20d | 0.0187 |
| SLV_ma_20d | 0.0156 |

### LSTM Return Forecaster

| Metric | Value |
|--------|-------|
| Best Test MSE | 2.42e-04 |
| Final Test R² | 0.0024 |
| Directional Accuracy | 53.8% |
| Parameters | 212,682 |
| Training Samples | 954 |
| Test Samples | 239 |
| Epochs Trained | 50 |

**Architecture:** 2-layer LSTM, 128 hidden units, 0.2 dropout, 60-day input sequences, simultaneous 10-asset output  
**Training:** Adam optimizer (lr=0.001), batch size 32, MSE loss, best-model checkpointing

The LSTM achieves 53.8% directional accuracy — marginally above chance and consistent with the return forecasting literature using sequential deep learning on diversified ETFs.

## Model Architecture Summary

| Component | Details | Training Hardware |
|-----------|---------|------------------|
| XGBoost | 100 estimators, max depth 5, lr 0.1, 130 features | NVIDIA H100 80GB HBM3 |
| LSTM | 2-layer, 128 hidden units, 0.2 dropout, 60-day sequences | NVIDIA H100 80GB HBM3 |
| FinBERT | ProsusAI/finbert (pretrained transformer) | Loaded from HuggingFace |
| Optimizer | CVXPY convex solver | CPU |

## Optimization Methods

The system supports four portfolio construction approaches:

**Maximum Sharpe Ratio:** Maximizes risk-adjusted return using mean-variance utility with configurable risk aversion (default: 2.5). The standard approach for growth-oriented allocations.

**Minimum Volatility:** Minimizes portfolio variance directly. Conservative approach for risk-averse mandates where drawdown protection takes priority over returns.

**Black-Litterman:** Combines market-implied equilibrium returns (reverse optimized from market cap weights) with user-specified views via Bayesian updating (tau=0.05). Produces more stable, less concentrated allocations than standard MVO.

**ML Enhanced:** Replaces historical expected returns with a 70/30 blend of historical and ML-predicted returns (XGBoost + LSTM average) before running mean-variance optimization.

## Technical Stack

- **Optimization:** CVXPY (convex portfolio optimization)
- **Machine Learning:** XGBoost, PyTorch (LSTM)
- **NLP/Sentiment:** Transformers (FinBERT — ProsusAI/finbert)
- **Factor Models:** statsmodels (OLS regression)
- **Risk Modeling:** scipy, numpy (VaR, CVaR, Monte Carlo)
- **Market Data:** yfinance (historical), Finnhub API (news), Alpha Vantage / Polygon (optional)
- **UI Framework:** Gradio
- **Visualization:** Plotly
- **Development:** Google Colab Pro with H100 GPU

## Setup and Usage

### Running in Google Colab

1. Clone this repository or download the notebook file
2. Upload to Google Colab
3. Select Runtime > Change runtime type > H100 GPU (or A100/T4 for free tier)
4. Run all cells sequentially

The notebook will automatically:
- Install required dependencies
- Download historical price data via yfinance
- Train XGBoost and LSTM models
- Compute factor loadings and Black-Litterman posteriors
- Launch a Gradio interface with a shareable link

### Running Locally

```bash
# Clone the repository
git clone https://github.com/SpencerCPurdy/AI_Powered_Multi_Asset_Portfolio_Optimizer.git
cd AI_Powered_Multi_Asset_Portfolio_Optimizer

# Install dependencies
pip install torch torchvision torchaudio transformers pandas numpy scipy scikit-learn xgboost cvxpy yfinance requests plotly gradio seaborn matplotlib statsmodels arch

# Run the application
python app.py
```

**Note:** Pre-trained model weights (`portfolio_models.zip`) must be present in the repository root. Training from scratch requires an H100/A100 GPU and approximately 10–15 minutes.

## Project Structure

```
├── app.py
├── portfolio_models.zip
├── training_metrics.json
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

The notebook contains the following components:

1. **Configuration & Setup**: `PortfolioConfig` dataclass with all hyperparameters documented
2. **Market Data Loader**: yfinance historical data with 130-feature engineering pipeline
3. **ML Models**: XGBoost and LSTM return predictors loaded from pre-trained weights
4. **Factor Model Engine**: Fama-French 5-factor OLS regression for each asset
5. **Portfolio Optimizer**: MPT, Black-Litterman, and ML-enhanced optimization via CVXPY
6. **Risk Engine**: VaR, CVaR, drawdown, Sharpe, Sortino, and Information ratio computation
7. **Monte Carlo Engine**: Cholesky-decomposed correlated path simulation
8. **Sentiment Engine**: FinBERT inference over Finnhub news feed
9. **Gradio Interface**: Seven-tab interactive application

## Key Implementation Details

- **Reproducibility:** All random seeds fixed to 42 across Python, NumPy, and PyTorch
- **Train/Test Split:** Fixed 80/20 chronological split; no look-ahead bias
- **ML Signal Blend:** 70% historical returns + 30% ML predictions at optimization time
- **Rebalancing Trigger:** 5% weight drift from target before rebalancing is recommended
- **Weight Constraints:** Minimum 0%, maximum 30% per asset; fully invested
- **VaR/CVaR:** 95% confidence level, 1-day horizon, both historical and parametric methods
- **Monte Carlo:** Cholesky decomposition for correlated return sampling; up to 50,000 paths

## API Integration

| Provider | Data Type | Requirement |
|----------|-----------|-------------|
| yfinance | Historical daily OHLCV | No key required |
| Finnhub | Company news (sentiment) | `FINNHUB_API_KEY` env variable |
| Alpha Vantage | Intraday data (optional) | `ALPHA_VANTAGE_API_KEY` env variable |
| Polygon.io | Real-time quotes (optional) | `POLYGON_API_KEY` env variable |

## Risk Parameter Reference

| Parameter | Value |
|-----------|-------|
| Risk-Free Rate | 4.0% |
| Risk Aversion (λ) | 2.5 |
| Black-Litterman tau | 0.05 |
| VaR / CVaR Confidence | 95% |
| Monte Carlo Simulations | Up to 50,000 paths |
| Rebalancing Threshold | 5% weight drift |
| Transaction Cost | 10 bps per trade |
| Sentiment Window | 30 days |
| Min Asset Weight | 0% |
| Max Asset Weight | 30% |

## Limitations and Known Issues

### Model Limitations
- XGBoost test R² of -3.13 and 38.4% directional accuracy confirm limited predictive power; its role is as a blended regularizing signal, not a primary alpha source
- LSTM test R² of 0.0024 reflects near-zero point forecast accuracy; 53.8% directional accuracy provides a marginal signal only
- Both models were trained on a specific 5-year market regime; out-of-sample generalization to future regimes is not guaranteed

### Financial Model Limitations
- Mean-variance optimization assumes normal return distributions; real returns exhibit fat tails and skewness
- Black-Litterman views are user-specified; the quality of output depends entirely on the quality of inputs
- Fama-French factor loadings estimated on historical data are subject to regime instability
- Covariance matrix estimation error affects all optimization approaches (the "error maximization" problem)
- Sentiment signals have unknown lag; news may already be priced in by the time it is scored

### Operational Limitations
- Optimization is single-period (myopic); dynamic multi-period optimization is not implemented
- No short-selling or derivatives
- Transaction cost model is simplified with no market impact component
- FinBERT inference at startup may be slow on free-tier CPU hardware

### Known Failure Modes
- Black swan events and regime shifts outside the training window
- Correlation structure breaks down during market crises, undermining diversification benefits
- Small changes in expected return estimates can produce large allocation shifts in unconstrained MVO
- Sentiment signals are subject to noise, manipulation, and event lag

## References

1. Markowitz (1952) — *Portfolio Selection*
2. Sharpe (1964) — *Capital Asset Pricing Model*
3. Black and Litterman (1992) — *Global Portfolio Optimization*
4. Fama and French (1993, 2015) — *Common Risk Factors; Five-Factor Model*
5. Gu, Kelly, Xiu (2020) — *Empirical Asset Pricing via Machine Learning*
6. Jorion (2006) — *Value at Risk*
7. Rockafellar and Uryasev (2000) — *Optimization of Conditional Value-at-Risk*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Spencer Purdy**  
GitHub: [@SpencerCPurdy](https://github.com/SpencerCPurdy)

---

*This is a portfolio project developed to demonstrate quantitative finance, portfolio optimization, machine learning, and production ML engineering. The system is designed for educational and demonstrational purposes only. Real investing involves significant financial risk. Past performance does not guarantee future results. Always consult with licensed financial professionals before making investment decisions.*
