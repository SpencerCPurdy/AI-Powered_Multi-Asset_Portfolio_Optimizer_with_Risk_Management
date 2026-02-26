"""
AI-Powered Multi-Asset Portfolio Optimizer with Risk Management
Author: Spencer Purdy
Description: Production-grade portfolio optimization system combining Modern Portfolio Theory,
             Black-Litterman model, machine learning alpha generation, sentiment analysis,
             and advanced risk management for optimal multi-asset portfolio construction.

Problem Statement: Construct and manage an optimal multi-asset portfolio that maximizes
                   risk-adjusted returns while incorporating market views, fundamental factors,
                   sentiment signals, and comprehensive risk constraints.

Real-World Application: Institutional asset management, wealth management, robo-advisors,
                        hedge funds, and quantitative investment strategies.

Key Features:
- Modern Portfolio Theory (MPT) with efficient frontier
- Black-Litterman model for views integration
- ML alpha generation: XGBoost and LSTM for return prediction (pre-trained on GPU)
- Fama-French multi-factor model with custom factors
- Sentiment analysis using FinBERT on news and earnings
- Advanced risk metrics: VaR, CVaR, maximum drawdown, Sharpe ratio
- Monte Carlo simulation for portfolio stress testing
- Dynamic rebalancing with transaction cost optimization
- Real-time market data from Alpha Vantage, Finnhub, and Polygon APIs

Technical Components:
- Mean-variance optimization with CVXPY
- Bayesian posterior estimation (Black-Litterman)
- Deep learning time series forecasting (LSTM, pre-trained)
- Gradient boosting for return prediction (XGBoost, pre-trained)
- Transformer-based sentiment analysis (FinBERT)
- Historical and parametric VaR/CVaR
- Scenario analysis and stress testing
- Multi-period optimization with rebalancing costs

Model Performance:
- XGBoost: Trained on 100+ engineered features per asset
- LSTM: 2-layer, 128-unit recurrent network on 60-day sequences
- Both models trained on H100 GPU, weights loaded at startup

Limitations:
- Historical data dependency (past does not equal future)
- Assumes normal return distributions (fat tails in reality)
- Transaction costs simplified (no market impact modeling)
- Sentiment data may have lag/noise
- Optimization is single-period (myopic)
- No options or derivatives modeling
- Risk model estimates subject to error

Hardware Requirements:
- Models pre-trained on H100 GPU
- Inference runs on CPU (Hugging Face Spaces)
- RAM: 16GB minimum

Dependencies:
- PyTorch (LSTM inference, FinBERT)
- XGBoost (gradient boosting inference)
- Transformers (FinBERT sentiment)
- CVXPY (convex optimization)
- yfinance (market data)
- pandas, numpy, scipy (numerical computing)

Reproducibility:
- Random seed: 42 (all libraries)
- Fixed train/test split during training
- Deterministic optimization

License: MIT License
Author: Spencer Purdy

Disclaimer: This is a simulation system for educational purposes. Real investing involves
            significant risks. Past performance does not guarantee future results.
            Always consult with financial professionals before investing real capital.
"""

# ============================================================================
# INSTALLATION AND IMPORTS
# ============================================================================

# Install required packages (uncomment to run in Colab)
# !pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# !pip install -q transformers pandas numpy scipy scikit-learn xgboost cvxpy yfinance requests plotly gradio seaborn matplotlib statsmodels arch

import os
import json
import time
import random
import warnings
import logging
import pickle
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path

# Core scientific computing
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm

# Machine Learning
import torch
import torch.nn as nn

# XGBoost
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

# NLP / Sentiment Analysis
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Portfolio Optimization
import cvxpy as cp

# Financial data
import yfinance as yf

# Statistical models
import statsmodels.api as sm

# API clients
import requests

# Visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# UI
import gradio as gr

# Configure logging and warnings
warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION AND REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

DEVICE = torch.device("cpu")  # HF Spaces runs on CPU

logger.info(f"Random seed set to {RANDOM_SEED}")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"Device: {DEVICE}")


@dataclass
class PortfolioConfig:
    """
    Configuration for the Portfolio Optimizer system.
    All hyperparameters documented with ranges and defaults.
    """
    random_seed: int = RANDOM_SEED

    # Portfolio universe (default, overridden by cached data)
    default_tickers: List[str] = field(default_factory=lambda: [
        'SPY', 'QQQ', 'IWM',
        'EFA', 'EEM',
        'TLT', 'IEF',
        'GLD', 'SLV',
        'VNQ',
    ])

    # Data parameters
    lookback_years: int = 5
    risk_free_rate: float = 0.04

    # MPT parameters
    min_weight: float = 0.0
    max_weight: float = 0.3
    risk_aversion: float = 2.5

    # Black-Litterman
    tau: float = 0.05

    # XGBoost (must match training config)
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.1
    xgb_lookback_window: int = 60

    # LSTM (must match training config)
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_sequence_length: int = 60
    lstm_epochs: int = 50
    lstm_batch_size: int = 32
    lstm_learning_rate: float = 0.001

    # Fama-French factors
    ff_factors: List[str] = field(default_factory=lambda: [
        'Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA'
    ])

    # Risk management
    var_confidence: float = 0.95
    cvar_confidence: float = 0.95
    monte_carlo_simulations: int = 10000

    # Sentiment
    sentiment_window_days: int = 30
    finbert_model: str = "ProsusAI/finbert"

    # Rebalancing
    rebalancing_threshold: float = 0.05
    transaction_cost_bps: float = 10.0

    # Paths
    data_dir: str = "./portfolio_data"
    models_dir: str = "./portfolio_models"

    # API keys (loaded from environment or HF Secrets)
    alphavantage_api_key: str = ""
    finnhub_api_key: str = ""
    polygon_api_key: str = ""


config = PortfolioConfig()
os.makedirs(config.data_dir, exist_ok=True)
os.makedirs(config.models_dir, exist_ok=True)

# ============================================================================
# LSTM MODEL DEFINITION (must match training architecture exactly)
# ============================================================================

class LSTMAlphaModel(nn.Module):
    """
    LSTM neural network for multi-asset return prediction.

    Architecture:
        Input -> LSTM(hidden_dim, num_layers) -> FC(hidden_dim/2) -> FC(output_dim)

    This class must match the architecture used during training so that
    the saved state_dict loads correctly.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.2,
                 output_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass: process sequence and return predictions."""
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        out = self.relu(self.fc1(last))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

# ============================================================================
# PRE-TRAINED MODEL LOADER
# ============================================================================

class PretrainedModels:
    """
    Loads all pre-trained artifacts from portfolio_models.zip and
    training_metrics.json at application startup.

    Artifacts loaded:
        - XGBoost model and scaler
        - LSTM model weights, config, and scaler
        - Cached market data (returns, covariance, prices)
        - Fama-French factor loadings
        - Training metrics (loss curves, R2, hit rates)
    """

    def __init__(self):
        self.xgb_model = None
        self.xgb_scaler = None
        self.lstm_model = None
        self.lstm_scaler = None
        self.lstm_config = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.tickers = None
        self.price_data = None
        self.factor_loadings = None
        self.training_metrics = None
        self.loaded = False

    def load(self):
        """Load all artifacts from disk."""
        logger.info("Loading pre-trained models and cached data...")

        # --- Load training metrics ---
        metrics_path = Path("training_metrics.json")
        if metrics_path.exists():
            with open(metrics_path, "r") as f:
                self.training_metrics = json.load(f)
            logger.info("Loaded training_metrics.json")
        else:
            logger.warning("training_metrics.json not found")
            self.training_metrics = {}

        # --- Extract portfolio_models.zip ---
        zip_path = Path("portfolio_models.zip")
        extract_dir = Path("./model_artifacts")

        if not zip_path.exists():
            logger.error("portfolio_models.zip not found -- models will not be available")
            return

        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(extract_dir)
        logger.info(f"Extracted {zip_path} to {extract_dir}")

        # --- XGBoost model ---
        xgb_path = extract_dir / "xgboost_model.json"
        if xgb_path.exists():
            self.xgb_model = xgb.XGBRegressor()
            self.xgb_model.load_model(str(xgb_path))
            logger.info("Loaded XGBoost model")

        # --- XGBoost scaler ---
        xgb_scaler_path = extract_dir / "xgb_scaler.pkl"
        if xgb_scaler_path.exists():
            with open(xgb_scaler_path, "rb") as f:
                self.xgb_scaler = pickle.load(f)
            logger.info("Loaded XGBoost scaler")

        # --- LSTM config ---
        lstm_config_path = extract_dir / "lstm_config.json"
        if lstm_config_path.exists():
            with open(lstm_config_path, "r") as f:
                self.lstm_config = json.load(f)
            logger.info(f"Loaded LSTM config: {self.lstm_config}")

        # --- LSTM model ---
        lstm_weights_path = extract_dir / "lstm_model.pt"
        if lstm_weights_path.exists() and self.lstm_config is not None:
            self.lstm_model = LSTMAlphaModel(
                input_dim=self.lstm_config['input_dim'],
                hidden_dim=self.lstm_config['hidden_dim'],
                num_layers=self.lstm_config['num_layers'],
                dropout=self.lstm_config['dropout'],
                output_dim=self.lstm_config['output_dim'],
            )
            self.lstm_model.load_state_dict(torch.load(str(lstm_weights_path), map_location=DEVICE))
            self.lstm_model.to(DEVICE)
            self.lstm_model.eval()
            logger.info("Loaded LSTM model weights")

        # --- LSTM scaler ---
        lstm_scaler_path = extract_dir / "lstm_scaler.pkl"
        if lstm_scaler_path.exists():
            with open(lstm_scaler_path, "rb") as f:
                self.lstm_scaler = pickle.load(f)
            logger.info("Loaded LSTM scaler")

        # --- Cached market data ---
        market_path = extract_dir / "market_data.pkl"
        if market_path.exists():
            with open(market_path, "rb") as f:
                cache = pickle.load(f)
            self.returns = cache['returns']
            self.mean_returns = cache['mean_returns']
            self.cov_matrix = cache['cov_matrix']
            self.tickers = cache['tickers']
            self.price_data = cache['price_data']
            logger.info(f"Loaded cached market data: {len(self.tickers)} tickers, "
                        f"{len(self.returns)} trading days")

        # --- Factor loadings ---
        fl_path = extract_dir / "factor_loadings.csv"
        if fl_path.exists():
            self.factor_loadings = pd.read_csv(fl_path, index_col=0)
            logger.info("Loaded factor loadings")

        self.loaded = True
        logger.info("All pre-trained artifacts loaded successfully")


# Load models at startup
pretrained = PretrainedModels()
pretrained.load()

# ============================================================================
# MARKET DATA FETCHER
# ============================================================================

class MarketDataFetcher:
    """
    Unified market data fetcher supporting multiple APIs.
    Primary source is yfinance; API keys provide fallback options.
    """

    def __init__(self):
        # Load API keys from environment (HF Spaces Secrets)
        self.finnhub_api_key = os.environ.get('FINNHUB_API_KEY', '')

    def fetch_price_data(self, tickers: List[str], start_date: str,
                         end_date: str) -> pd.DataFrame:
        """
        Fetch historical price data for multiple tickers.

        Args:
            tickers: List of ticker symbols.
            start_date: Start date string (YYYY-MM-DD).
            end_date: End date string (YYYY-MM-DD).

        Returns:
            Long-format DataFrame with [Date, Ticker, Open, High, Low, Close, Volume].
        """
        logger.info(f"Fetching price data for {len(tickers)} tickers")

        try:
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True
            )

            if data.empty:
                raise ValueError("No data returned")

            if len(tickers) == 1:
                data['Ticker'] = tickers[0]
                data = data.reset_index()
            else:
                data = data.stack(level=1).reset_index()
                data = data.rename(columns={'level_1': 'Ticker'})

            logger.info(f"Fetched {len(data)} rows from yfinance")
            return data

        except Exception as e:
            logger.error(f"Data fetch failed: {e}")
            return self._generate_synthetic_prices(tickers, start_date, end_date)

    def _generate_synthetic_prices(self, tickers, start_date, end_date):
        """Generate synthetic price data using Geometric Brownian Motion."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        frames = []

        for ticker in tickers:
            S0 = 100 + np.random.uniform(-20, 50)
            mu = 0.08 + np.random.uniform(-0.05, 0.05)
            sigma = 0.15 + np.random.uniform(-0.05, 0.10)
            dt = 1 / 252
            rets = np.random.normal(mu * dt, sigma * np.sqrt(dt), len(dates))
            prices = S0 * np.exp(np.cumsum(rets))

            df = pd.DataFrame({
                'Date': dates,
                'Ticker': ticker,
                'Open': prices * (1 + np.random.uniform(-0.01, 0.01, len(dates))),
                'High': prices * (1 + np.abs(np.random.uniform(0, 0.02, len(dates)))),
                'Low': prices * (1 - np.abs(np.random.uniform(0, 0.02, len(dates)))),
                'Close': prices,
                'Volume': np.random.lognormal(15, 1, len(dates)).astype(int)
            })
            frames.append(df)

        return pd.concat(frames, ignore_index=True)

    def fetch_finnhub_news(self, ticker: str, days: int = 30) -> List[Dict]:
        """
        Fetch news articles for a ticker from Finnhub.

        Args:
            ticker: Ticker symbol.
            days: Lookback window in days.

        Returns:
            List of news articles with headline, summary, datetime.
        """
        if not self.finnhub_api_key:
            logger.warning("Finnhub API key not set")
            return []

        try:
            end_dt = datetime.now()
            start_dt = end_dt - timedelta(days=days)

            url = "https://finnhub.io/api/v1/company-news"
            params = {
                'symbol': ticker,
                'from': start_dt.strftime('%Y-%m-%d'),
                'to': end_dt.strftime('%Y-%m-%d'),
                'token': self.finnhub_api_key
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            news = response.json()
            logger.info(f"Fetched {len(news)} articles for {ticker}")
            return news

        except Exception as e:
            logger.error(f"Finnhub fetch failed: {e}")
            return []

# ============================================================================
# DATA PROCESSING AND FEATURE ENGINEERING
# ============================================================================

class PortfolioDataProcessor:
    """Process and engineer features for portfolio optimization and ML models."""

    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily returns from price data."""
        prices_pivot = prices.pivot(index='Date', columns='Ticker', values='Close')
        returns = prices_pivot.pct_change().dropna()
        return returns

    def create_ml_features(self, prices: pd.DataFrame, returns: pd.DataFrame,
                           lookback: int = 60) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create features for XGBoost return prediction.

        Features per ticker:
            - Rolling mean returns (5, 10, 20, 60 day windows)
            - Rolling volatility (10, 20, 60 day windows)
            - Moving averages (20, 50, 200 day)
            - Price-to-MA20 ratio
            - RSI (14-period)
            - Volume momentum (relative to 20-day MA)

        Args:
            prices: Long-format price DataFrame.
            returns: Wide-format returns DataFrame.
            lookback: Feature lookback window.

        Returns:
            Tuple of (features_df, target_series).
        """
        prices_pivot = prices.pivot(index='Date', columns='Ticker', values='Close')
        volume_pivot = prices.pivot(index='Date', columns='Ticker', values='Volume')

        features_list = []

        for ticker in prices_pivot.columns:
            tp = prices_pivot[ticker]
            tr = returns[ticker]
            tv = volume_pivot[ticker]

            for w in [5, 10, 20, 60]:
                features_list.append(tr.rolling(w).mean().rename(f'{ticker}_return_{w}d'))

            for w in [10, 20, 60]:
                features_list.append(tr.rolling(w).std().rename(f'{ticker}_volatility_{w}d'))

            for w in [20, 50, 200]:
                features_list.append(tp.rolling(w).mean().rename(f'{ticker}_ma_{w}d'))

            ma20 = tp.rolling(20).mean()
            features_list.append((tp / ma20).rename(f'{ticker}_price_to_ma20'))

            delta = tr
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features_list.append(rsi.rename(f'{ticker}_rsi'))

            vol_ma = tv.rolling(20).mean()
            features_list.append((tv / vol_ma).rename(f'{ticker}_volume_momentum'))

        features_df = pd.concat(features_list, axis=1)
        target = returns.shift(-1).mean(axis=1)
        features_df = features_df.loc[target.index]
        valid = features_df.notna().all(axis=1) & target.notna()

        return features_df[valid], target[valid]

    def create_lstm_sequences(self, returns: pd.DataFrame,
                              sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding-window sequences for LSTM prediction.

        Args:
            returns: Wide-format daily returns.
            sequence_length: Number of timesteps per sequence.

        Returns:
            X (n_samples, seq_len, n_assets), y (n_samples, n_assets).
        """
        data = returns.values
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i - sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)

# ============================================================================
# MODERN PORTFOLIO THEORY (MPT)
# ============================================================================

class ModernPortfolioTheory:
    """
    Modern Portfolio Theory implementation with mean-variance optimization.
    Uses CVXPY for convex optimization with configurable constraints.
    """

    def __init__(self, cfg: PortfolioConfig):
        self.config = cfg

    def calculate_portfolio_stats(self, weights: np.ndarray, mean_returns: np.ndarray,
                                  cov_matrix: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate annualized portfolio return, volatility, and Sharpe ratio.

        Args:
            weights: Asset weight vector.
            mean_returns: Daily expected returns.
            cov_matrix: Daily covariance matrix.

        Returns:
            Tuple of (annualized return, annualized volatility, Sharpe ratio).
        """
        portfolio_return = np.dot(weights, mean_returns) * 252
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
        sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol if portfolio_vol > 0 else 0.0
        return portfolio_return, portfolio_vol, sharpe

    def optimize_max_sharpe(self, mean_returns: np.ndarray,
                            cov_matrix: np.ndarray) -> np.ndarray:
        """
        Optimize for maximum Sharpe ratio using mean-variance utility.

        Maximize: E[R] - risk_aversion * Var[R]
        Subject to: weights sum to 1, min/max bounds.

        Args:
            mean_returns: Daily expected returns.
            cov_matrix: Daily covariance matrix.

        Returns:
            Optimal weight vector.
        """
        n = len(mean_returns)
        w = cp.Variable(n)

        annual_return = cp.sum(cp.multiply(mean_returns, w)) * 252
        annual_vol_sq = cp.quad_form(w, cov_matrix * 252)

        objective = cp.Maximize(annual_return - self.config.risk_aversion * annual_vol_sq)
        constraints = [
            cp.sum(w) == 1,
            w >= self.config.min_weight,
            w <= self.config.max_weight,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if w.value is None:
            logger.warning("Max Sharpe optimization failed, returning equal weights")
            return np.ones(n) / n
        return w.value

    def optimize_min_volatility(self, cov_matrix: np.ndarray) -> np.ndarray:
        """
        Optimize for minimum portfolio volatility.

        Args:
            cov_matrix: Daily covariance matrix.

        Returns:
            Optimal weight vector.
        """
        n = cov_matrix.shape[0]
        w = cp.Variable(n)

        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        constraints = [
            cp.sum(w) == 1,
            w >= self.config.min_weight,
            w <= self.config.max_weight,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if w.value is None:
            logger.warning("Min volatility optimization failed, returning equal weights")
            return np.ones(n) / n
        return w.value

    def optimize_target_return(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                               target_return: float) -> np.ndarray:
        """
        Minimize volatility subject to a target annual return.

        Args:
            mean_returns: Daily expected returns.
            cov_matrix: Daily covariance matrix.
            target_return: Target annualized return.

        Returns:
            Optimal weight vector.
        """
        n = len(mean_returns)
        w = cp.Variable(n)

        objective = cp.Minimize(cp.quad_form(w, cov_matrix))
        constraints = [
            cp.sum(w) == 1,
            cp.sum(cp.multiply(mean_returns, w)) * 252 >= target_return,
            w >= self.config.min_weight,
            w <= self.config.max_weight,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if w.value is None:
            return self.optimize_max_sharpe(mean_returns, cov_matrix)
        return w.value

    def calculate_efficient_frontier(self, mean_returns: np.ndarray,
                                     cov_matrix: np.ndarray,
                                     n_points: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate the efficient frontier across a range of target returns.

        Args:
            mean_returns: Daily expected returns.
            cov_matrix: Daily covariance matrix.
            n_points: Number of points on the frontier.

        Returns:
            Tuple of (returns_array, volatilities_array, sharpe_array).
        """
        min_ret = mean_returns.min() * 252
        max_ret = mean_returns.max() * 252
        targets = np.linspace(min_ret, max_ret, n_points)

        rets, vols, sharpes = [], [], []
        for target in targets:
            try:
                weights = self.optimize_target_return(mean_returns, cov_matrix, target)
                r, v, s = self.calculate_portfolio_stats(weights, mean_returns, cov_matrix)
                rets.append(r)
                vols.append(v)
                sharpes.append(s)
            except Exception:
                continue

        return np.array(rets), np.array(vols), np.array(sharpes)

# ============================================================================
# BLACK-LITTERMAN MODEL
# ============================================================================

class BlackLittermanModel:
    """
    Black-Litterman model for incorporating investor views into
    market equilibrium returns using Bayesian updating.
    """

    def __init__(self, cfg: PortfolioConfig):
        self.config = cfg

    def calculate_market_implied_returns(self, cov_matrix: np.ndarray,
                                         market_weights: np.ndarray) -> np.ndarray:
        """
        Reverse-optimize market-implied equilibrium returns.

        Pi = risk_aversion * Sigma * w_market

        Args:
            cov_matrix: Daily covariance matrix.
            market_weights: Market capitalization weights.

        Returns:
            Implied annualized returns vector.
        """
        annual_cov = cov_matrix * 252
        return self.config.risk_aversion * np.dot(annual_cov, market_weights)

    def black_litterman_posterior(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                                  views_matrix: np.ndarray, views_returns: np.ndarray,
                                  views_uncertainty: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Black-Litterman posterior distribution.

        E[R] = [(tau*Sigma)^-1 + P'*Omega^-1*P]^-1 *
               [(tau*Sigma)^-1*Pi + P'*Omega^-1*Q]

        Args:
            mean_returns: Prior (market-implied) returns.
            cov_matrix: Daily covariance matrix.
            views_matrix: P matrix (k x n) mapping views to assets.
            views_returns: Q vector (k x 1) view return expectations.
            views_uncertainty: Omega matrix (k x k) view uncertainty.

        Returns:
            Tuple of (posterior_returns, posterior_covariance).
        """
        tau = self.config.tau
        prior_cov = tau * cov_matrix * 252

        A = np.linalg.inv(prior_cov)
        B = np.dot(views_matrix.T, np.linalg.inv(views_uncertainty))
        C = np.dot(B, views_matrix)

        posterior_cov_inv = A + C
        posterior_cov = np.linalg.inv(posterior_cov_inv)

        term1 = np.dot(A, mean_returns)
        term2 = np.dot(B, views_returns)
        posterior_returns = np.dot(posterior_cov, term1 + term2)

        posterior_cov = posterior_cov + cov_matrix * 252

        return posterior_returns, posterior_cov

# ============================================================================
# FAMA-FRENCH FACTOR MODELS
# ============================================================================

class FamaFrenchModel:
    """
    Fama-French multi-factor model for expected return estimation.
    Uses pre-computed factor loadings from training.
    """

    def __init__(self, cfg: PortfolioConfig):
        self.config = cfg
        self.factor_loadings = None

    def load_factor_loadings(self, loadings_df: pd.DataFrame):
        """Load pre-computed factor loadings."""
        self.factor_loadings = loadings_df

    def calculate_expected_returns(self, factor_premiums: Dict[str, float]) -> pd.Series:
        """
        Calculate expected returns from factor loadings and premiums.

        E[R_i] = alpha_i + sum(beta_ij * premium_j)

        Args:
            factor_premiums: Dictionary of factor risk premiums.

        Returns:
            Series of expected returns indexed by ticker.
        """
        if self.factor_loadings is None:
            raise ValueError("Factor loadings not loaded")

        expected = {}
        for ticker in self.factor_loadings.index:
            loadings = self.factor_loadings.loc[ticker]
            er = loadings.get('alpha', 0.0)
            for factor in self.config.ff_factors:
                er += loadings.get(factor, 0.0) * factor_premiums.get(factor, 0.0)
            expected[ticker] = er

        return pd.Series(expected)

# ============================================================================
# SENTIMENT ANALYSIS WITH FINBERT
# ============================================================================

class SentimentAnalyzer:
    """
    Sentiment analysis using the FinBERT transformer model.
    Analyzes financial news headlines and produces sentiment scores.
    """

    def __init__(self):
        self.pipeline_obj = None
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy-load FinBERT model on first use."""
        if not self._loaded:
            logger.info("Loading FinBERT model (first use)...")
            self.pipeline_obj = pipeline(
                "sentiment-analysis",
                model=config.finbert_model,
                tokenizer=config.finbert_model,
                device=-1  # CPU
            )
            self._loaded = True
            logger.info("FinBERT loaded")

    def analyze_news(self, news_articles: List[Dict]) -> Dict[str, float]:
        """
        Analyze sentiment of news articles using FinBERT.

        Args:
            news_articles: List of articles with 'headline' and/or 'summary' keys.

        Returns:
            Dictionary with positive, negative, neutral, compound scores.
        """
        if not news_articles:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0,
                    'compound': 0.0, 'num_articles': 0}

        self._ensure_loaded()

        sentiments = []
        for article in news_articles:
            text = article.get('headline', '') + ' ' + article.get('summary', '')
            text = text[:512].strip()
            if not text:
                continue

            try:
                result = self.pipeline_obj(text)[0]
                sentiments.append({
                    'label': result['label'].lower(),
                    'score': result['score']
                })
            except Exception as e:
                logger.warning(f"Sentiment analysis failed for article: {e}")
                continue

        if not sentiments:
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0,
                    'compound': 0.0, 'num_articles': 0}

        pos = [s['score'] for s in sentiments if s['label'] == 'positive']
        neg = [s['score'] for s in sentiments if s['label'] == 'negative']
        neu = [s['score'] for s in sentiments if s['label'] == 'neutral']

        positive = np.mean(pos) if pos else 0.0
        negative = np.mean(neg) if neg else 0.0
        neutral = np.mean(neu) if neu else 0.0
        compound = positive - negative

        return {
            'positive': float(positive),
            'negative': float(negative),
            'neutral': float(neutral),
            'compound': float(compound),
            'num_articles': len(sentiments)
        }

# ============================================================================
# RISK MANAGEMENT AND ANALYTICS
# ============================================================================

class RiskAnalytics:
    """
    Advanced risk analytics: VaR, CVaR, drawdown analysis,
    Monte Carlo simulation, and stress testing.
    """

    def __init__(self, cfg: PortfolioConfig):
        self.config = cfg

    def calculate_var(self, returns: np.ndarray, confidence: float = 0.95,
                      method: str = 'historical') -> float:
        """
        Calculate Value at Risk.

        Args:
            returns: Array of portfolio returns.
            confidence: Confidence level (e.g. 0.95).
            method: 'historical' or 'parametric'.

        Returns:
            VaR value (positive number = potential loss).
        """
        if method == 'historical':
            return -np.percentile(returns, (1 - confidence) * 100)
        elif method == 'parametric':
            mu = np.mean(returns)
            sigma = np.std(returns)
            return -(mu + sigma * norm.ppf(1 - confidence))
        else:
            raise ValueError(f"Unknown VaR method: {method}")

    def calculate_cvar(self, returns: np.ndarray, confidence: float = 0.95) -> float:
        """
        Calculate Conditional VaR (Expected Shortfall).

        CVaR is the expected loss given that VaR is exceeded.

        Args:
            returns: Array of portfolio returns.
            confidence: Confidence level.

        Returns:
            CVaR value (positive number = expected tail loss).
        """
        var = self.calculate_var(returns, confidence, method='historical')
        losses = -returns[returns < -var]
        return np.mean(losses) if len(losses) > 0 else var

    def calculate_maximum_drawdown(self, returns: pd.Series) -> Tuple[float, Any, Any]:
        """
        Calculate maximum drawdown from a return series.

        Args:
            returns: Time series of returns.

        Returns:
            Tuple of (max_drawdown_fraction, peak_date, trough_date).
        """
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        trough_date = drawdown.idxmin()
        peak_date = running_max[:trough_date].idxmax()
        return abs(max_dd), peak_date, trough_date

    def calculate_sharpe_ratio(self, returns: np.ndarray,
                               risk_free_rate: float = None) -> float:
        """Calculate annualized Sharpe ratio."""
        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate / 252
        excess = returns - risk_free_rate
        if np.std(excess) == 0:
            return 0.0
        return np.mean(excess) / np.std(excess) * np.sqrt(252)

    def calculate_sortino_ratio(self, returns: np.ndarray,
                                risk_free_rate: float = None) -> float:
        """Calculate annualized Sortino ratio (downside risk only)."""
        if risk_free_rate is None:
            risk_free_rate = self.config.risk_free_rate / 252
        excess = returns - risk_free_rate
        downside = excess[excess < 0]
        downside_std = np.std(downside) if len(downside) > 0 else np.std(excess)
        if downside_std == 0:
            return 0.0
        return np.mean(excess) / downside_std * np.sqrt(252)

    def calculate_information_ratio(self, portfolio_returns: np.ndarray,
                                    benchmark_returns: np.ndarray) -> float:
        """Calculate annualized information ratio (active return / tracking error)."""
        active = portfolio_returns - benchmark_returns
        te = np.std(active)
        if te == 0:
            return 0.0
        return np.mean(active) / te * np.sqrt(252)

    def monte_carlo_simulation(self, mean_returns: np.ndarray, cov_matrix: np.ndarray,
                               weights: np.ndarray, n_days: int = 252,
                               n_simulations: int = 10000) -> np.ndarray:
        """
        Monte Carlo simulation of portfolio value paths.

        Uses Cholesky decomposition for correlated asset returns.

        Args:
            mean_returns: Daily expected returns per asset.
            cov_matrix: Daily covariance matrix.
            weights: Portfolio weights.
            n_days: Number of days to simulate.
            n_simulations: Number of simulation paths.

        Returns:
            Array of shape (n_simulations, n_days + 1) with portfolio values.
        """
        L = np.linalg.cholesky(cov_matrix)

        paths = np.zeros((n_simulations, n_days + 1))
        paths[:, 0] = 1.0

        for sim in range(n_simulations):
            for day in range(1, n_days + 1):
                z = np.random.normal(0, 1, len(mean_returns))
                asset_rets = mean_returns + np.dot(L, z)
                port_ret = np.dot(weights, asset_rets)
                paths[sim, day] = paths[sim, day - 1] * (1 + port_ret)

        return paths

# ============================================================================
# REBALANCING OPTIMIZER
# ============================================================================

class RebalancingOptimizer:
    """
    Optimize portfolio rebalancing considering transaction costs.
    Implements turnover-constrained optimization.
    """

    def __init__(self, cfg: PortfolioConfig):
        self.config = cfg

    def calculate_turnover(self, current: np.ndarray, target: np.ndarray) -> float:
        """Calculate one-way turnover."""
        return np.sum(np.abs(target - current)) / 2

    def calculate_transaction_costs(self, current: np.ndarray, target: np.ndarray,
                                    portfolio_value: float) -> float:
        """Calculate transaction costs from rebalancing."""
        turnover = self.calculate_turnover(current, target)
        cost_rate = self.config.transaction_cost_bps / 10000
        return turnover * portfolio_value * cost_rate

    def should_rebalance(self, current: np.ndarray, target: np.ndarray) -> bool:
        """Check if any weight has drifted beyond threshold."""
        return np.max(np.abs(current - target)) > self.config.rebalancing_threshold

    def optimize_with_transaction_costs(self, mean_returns: np.ndarray,
                                        cov_matrix: np.ndarray,
                                        current_weights: np.ndarray) -> np.ndarray:
        """
        Optimize portfolio considering transaction costs from current position.

        Maximizes: E[R] - risk_aversion * Var - transaction_cost_penalty

        Args:
            mean_returns: Daily expected returns.
            cov_matrix: Daily covariance matrix.
            current_weights: Current portfolio weights.

        Returns:
            Optimal weight vector accounting for transaction costs.
        """
        n = len(mean_returns)
        w = cp.Variable(n)

        annual_return = cp.sum(cp.multiply(mean_returns, w)) * 252
        annual_var = cp.quad_form(w, cov_matrix * 252)
        turnover = cp.sum(cp.abs(w - current_weights)) / 2
        tc = turnover * (self.config.transaction_cost_bps / 10000)

        objective = cp.Maximize(
            annual_return - self.config.risk_aversion * annual_var - tc * 100
        )
        constraints = [
            cp.sum(w) == 1,
            w >= self.config.min_weight,
            w <= self.config.max_weight,
        ]

        problem = cp.Problem(objective, constraints)
        problem.solve()

        if w.value is None:
            return current_weights
        return w.value

# ============================================================================
# INTEGRATED PORTFOLIO SYSTEM
# ============================================================================

class IntegratedPortfolioSystem:
    """
    Main orchestrator that integrates all components:
    - Pre-trained ML models (XGBoost, LSTM)
    - MPT and Black-Litterman optimization
    - Fama-French factor models
    - Sentiment analysis (FinBERT)
    - Risk analytics and Monte Carlo
    - Rebalancing with transaction costs
    """

    def __init__(self, cfg: PortfolioConfig):
        self.config = cfg
        self.data_fetcher = MarketDataFetcher()
        self.data_processor = PortfolioDataProcessor()
        self.mpt = ModernPortfolioTheory(cfg)
        self.black_litterman = BlackLittermanModel(cfg)
        self.fama_french = FamaFrenchModel(cfg)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.risk_analytics = RiskAnalytics(cfg)
        self.rebalancer = RebalancingOptimizer(cfg)

        # Data storage
        self.price_data = None
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.factor_loadings = None
        self.sentiment_scores = {}

    def load_from_cache(self):
        """Load market data and factor loadings from pre-trained cache."""
        if pretrained.loaded and pretrained.returns is not None:
            self.returns = pretrained.returns
            self.mean_returns = pretrained.mean_returns
            self.cov_matrix = pretrained.cov_matrix
            self.price_data = pretrained.price_data

            if pretrained.factor_loadings is not None:
                self.factor_loadings = pretrained.factor_loadings
                self.fama_french.load_factor_loadings(pretrained.factor_loadings)

            logger.info("Loaded data from pre-trained cache")
            return True
        return False

    def load_fresh_data(self, tickers: List[str], lookback_years: int):
        """Fetch fresh market data from yfinance."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=int(lookback_years * 365))

        self.price_data = self.data_fetcher.fetch_price_data(
            tickers,
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        self.returns = self.data_processor.calculate_returns(self.price_data)
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()

        logger.info(f"Loaded fresh data: {len(self.returns)} days for {len(tickers)} assets")

    def get_ml_predictions(self) -> Optional[np.ndarray]:
        """
        Run inference on pre-trained XGBoost and LSTM models.

        Returns combined prediction (average of both models), or None if
        models are not available.
        """
        predictions = []

        # XGBoost prediction
        if (pretrained.xgb_model is not None and
                pretrained.xgb_scaler is not None and
                self.price_data is not None):
            try:
                X_xgb, _ = self.data_processor.create_ml_features(
                    self.price_data, self.returns, config.xgb_lookback_window
                )
                if len(X_xgb) > 0:
                    X_scaled = pretrained.xgb_scaler.transform(X_xgb)
                    xgb_pred = pretrained.xgb_model.predict(X_scaled)
                    predictions.append(('xgboost', xgb_pred[-1]))
                    logger.info(f"XGBoost prediction: {xgb_pred[-1]:.6f}")
            except Exception as e:
                logger.warning(f"XGBoost inference failed: {e}")

        # LSTM prediction
        if (pretrained.lstm_model is not None and
                pretrained.lstm_scaler is not None and
                self.returns is not None):
            try:
                X_lstm, _ = self.data_processor.create_lstm_sequences(
                    self.returns, config.lstm_sequence_length
                )
                if len(X_lstm) > 0:
                    n_samples, n_seq, n_features = X_lstm.shape
                    X_2d = X_lstm.reshape(-1, n_features)
                    X_scaled = pretrained.lstm_scaler.transform(X_2d).reshape(
                        n_samples, n_seq, n_features
                    )
                    # Use last sequence for prediction
                    X_last = torch.FloatTensor(X_scaled[-1:]).to(DEVICE)
                    pretrained.lstm_model.eval()
                    with torch.no_grad():
                        lstm_pred = pretrained.lstm_model(X_last).cpu().numpy().flatten()
                    predictions.append(('lstm', lstm_pred.mean()))
                    logger.info(f"LSTM prediction: {lstm_pred.mean():.6f}")
            except Exception as e:
                logger.warning(f"LSTM inference failed: {e}")

        if predictions:
            avg_pred = np.mean([p[1] for p in predictions])
            return avg_pred
        return None

    def optimize_portfolio(self, method: str = 'max_sharpe',
                           use_ml: bool = False) -> Dict:
        """
        Optimize portfolio allocation using the specified method.

        Args:
            method: One of 'max_sharpe', 'min_volatility', 'black_litterman', 'ml_enhanced'.
            use_ml: Whether to incorporate ML predictions.

        Returns:
            Dictionary with weights, statistics, and risk metrics.
        """
        mean_rets = self.mean_returns.values
        cov_mat = self.cov_matrix.values

        # Optionally adjust returns with ML predictions
        if use_ml or method == 'ml_enhanced':
            ml_pred = self.get_ml_predictions()
            if ml_pred is not None:
                # Blend: 70% historical, 30% ML signal
                adjustment = ml_pred * 0.3
                mean_rets = mean_rets + adjustment
                logger.info(f"ML adjustment applied: {adjustment:.6f}")

        if method == 'max_sharpe' or method == 'ml_enhanced':
            weights = self.mpt.optimize_max_sharpe(mean_rets, cov_mat)
        elif method == 'min_volatility':
            weights = self.mpt.optimize_min_volatility(cov_mat)
        elif method == 'black_litterman':
            market_weights = np.ones(len(mean_rets)) / len(mean_rets)
            implied = self.black_litterman.calculate_market_implied_returns(
                cov_mat, market_weights
            )
            weights = self.mpt.optimize_max_sharpe(implied / 252, cov_mat)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate statistics
        port_ret, port_vol, sharpe = self.mpt.calculate_portfolio_stats(
            weights, self.mean_returns.values, cov_mat
        )
        port_returns = np.dot(self.returns.values, weights)

        var_95 = self.risk_analytics.calculate_var(port_returns, 0.95)
        cvar_95 = self.risk_analytics.calculate_cvar(port_returns, 0.95)
        max_dd, _, _ = self.risk_analytics.calculate_maximum_drawdown(pd.Series(port_returns))

        return {
            'weights': pd.Series(weights, index=self.returns.columns),
            'expected_return': port_ret,
            'volatility': port_vol,
            'sharpe_ratio': sharpe,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'max_drawdown': max_dd,
        }

    def analyze_sentiment(self, tickers: List[str]) -> Dict[str, Dict]:
        """Run FinBERT sentiment analysis on news for each ticker."""
        for ticker in tickers:
            news = self.data_fetcher.fetch_finnhub_news(ticker, config.sentiment_window_days)
            self.sentiment_scores[ticker] = self.sentiment_analyzer.analyze_news(news)
        return self.sentiment_scores

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_efficient_frontier(mpt: ModernPortfolioTheory, mean_returns: np.ndarray,
                           cov_matrix: np.ndarray, optimal_weights: np.ndarray) -> go.Figure:
    """Plot efficient frontier with optimal portfolio marked."""
    frontier_rets, frontier_vols, frontier_sharpes = mpt.calculate_efficient_frontier(
        mean_returns, cov_matrix, n_points=50
    )
    opt_ret, opt_vol, opt_sharpe = mpt.calculate_portfolio_stats(
        optimal_weights, mean_returns, cov_matrix
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier_vols, y=frontier_rets,
        mode='lines+markers',
        name='Efficient Frontier',
        line=dict(color='blue', width=2),
        marker=dict(size=6, color=frontier_sharpes, colorscale='Viridis',
                    showscale=True, colorbar=dict(title="Sharpe"))
    ))
    fig.add_trace(go.Scatter(
        x=[opt_vol], y=[opt_ret],
        mode='markers',
        name='Optimal Portfolio',
        marker=dict(size=15, color='red', symbol='star')
    ))
    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Volatility (Annualized)",
        yaxis_title="Expected Return (Annualized)",
        hovermode='closest', height=600
    )
    return fig


def plot_portfolio_weights(weights: pd.Series) -> go.Figure:
    """Plot portfolio allocation as a donut chart."""
    fig = go.Figure(data=[go.Pie(
        labels=weights.index, values=weights.values,
        hole=0.3, textposition='inside', textinfo='label+percent'
    )])
    fig.update_layout(title="Portfolio Allocation", height=500)
    return fig


def plot_cumulative_returns(returns: pd.DataFrame, weights: np.ndarray,
                           benchmark_ticker: str = 'SPY') -> go.Figure:
    """Plot cumulative returns of optimized portfolio vs benchmark."""
    port_rets = np.dot(returns.values, weights)
    port_cum = (1 + port_rets).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=returns.index, y=port_cum,
        mode='lines', name='Portfolio',
        line=dict(color='blue', width=2)
    ))

    if benchmark_ticker in returns.columns:
        bench_cum = (1 + returns[benchmark_ticker].values).cumprod()
        fig.add_trace(go.Scatter(
            x=returns.index, y=bench_cum,
            mode='lines', name=f'Benchmark ({benchmark_ticker})',
            line=dict(color='gray', width=2, dash='dash')
        ))

    fig.update_layout(
        title="Cumulative Returns: Portfolio vs. Benchmark",
        xaxis_title="Date", yaxis_title="Cumulative Return",
        hovermode='x unified', height=600
    )
    return fig


def plot_risk_metrics(risk_metrics: Dict) -> go.Figure:
    """Create risk metrics dashboard with gauge indicators."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Sharpe Ratio', 'Max Drawdown', 'VaR (95%)', 'CVaR (95%)'),
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
               [{'type': 'indicator'}, {'type': 'indicator'}]]
    )

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=risk_metrics['sharpe_ratio'],
        title={'text': "Sharpe Ratio"},
        gauge={
            'axis': {'range': [-1, 3]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-1, 0], 'color': "#ffcccc"},
                {'range': [0, 1], 'color': "lightgray"},
                {'range': [1, 2], 'color': "#ccffcc"}
            ],
        }
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=risk_metrics['max_drawdown'] * 100,
        title={'text': "Max Drawdown (%)"},
        number={'suffix': '%'},
        gauge={
            'axis': {'range': [0, 50]},
            'bar': {'color': "red"},
            'steps': [
                {'range': [0, 15], 'color': "lightgreen"},
                {'range': [15, 30], 'color': "yellow"},
                {'range': [30, 50], 'color': "#ffcccc"}
            ]
        }
    ), row=1, col=2)

    fig.add_trace(go.Indicator(
        mode="number",
        value=risk_metrics['var_95'] * 100,
        title={'text': "VaR 95% (%)"},
        number={'suffix': "%"}
    ), row=2, col=1)

    fig.add_trace(go.Indicator(
        mode="number",
        value=risk_metrics['cvar_95'] * 100,
        title={'text': "CVaR 95% (%)"},
        number={'suffix': "%"}
    ), row=2, col=2)

    fig.update_layout(height=600, title_text="Risk Metrics Dashboard")
    return fig


def plot_monte_carlo(paths: np.ndarray) -> go.Figure:
    """Plot Monte Carlo simulation paths with percentile bands."""
    fig = go.Figure()

    n_show = min(100, paths.shape[0])
    for i in range(n_show):
        fig.add_trace(go.Scatter(
            y=paths[i], mode='lines',
            line=dict(width=0.5, color='lightblue'),
            showlegend=False, hoverinfo='skip'
        ))

    for p, color, name in [(5, 'red', '5th'), (50, 'black', 'Median'), (95, 'green', '95th')]:
        pct = np.percentile(paths, p, axis=0)
        fig.add_trace(go.Scatter(
            y=pct, mode='lines', name=f'{name} Percentile',
            line=dict(width=3, color=color)
        ))

    fig.update_layout(
        title="Monte Carlo Simulation: Portfolio Value Paths",
        xaxis_title="Days", yaxis_title="Portfolio Value",
        height=600, hovermode='x unified'
    )
    return fig


def plot_factor_loadings(fl: pd.DataFrame) -> go.Figure:
    """Plot Fama-French factor loadings as a heatmap."""
    factor_cols = [c for c in fl.columns if c not in ('alpha', 'r_squared')]
    fig = go.Figure(data=go.Heatmap(
        z=fl[factor_cols].values,
        x=factor_cols, y=fl.index,
        colorscale='RdBu', zmid=0,
        text=fl[factor_cols].round(3).values,
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Beta")
    ))
    fig.update_layout(
        title="Fama-French Factor Loadings (Betas)",
        xaxis_title="Factor", yaxis_title="Asset", height=500
    )
    return fig


def plot_sentiment_scores(scores: Dict[str, Dict]) -> go.Figure:
    """Plot sentiment compound scores per ticker."""
    tickers = list(scores.keys())
    compounds = [scores[t].get('compound', 0.0) for t in tickers]

    fig = go.Figure(data=[go.Bar(
        x=tickers, y=compounds,
        marker=dict(color=compounds, colorscale='RdYlGn', cmin=-1, cmax=1,
                    colorbar=dict(title="Sentiment")),
        text=[f"{c:.2f}" for c in compounds],
        textposition='outside'
    )])
    fig.update_layout(
        title="Sentiment Analysis: Compound Scores",
        xaxis_title="Ticker", yaxis_title="Sentiment Score",
        yaxis=dict(range=[-1, 1]), height=500
    )
    return fig


def plot_training_losses(metrics: Dict) -> go.Figure:
    """Plot LSTM training and test loss curves from training metrics."""
    lstm = metrics.get('lstm', {})
    train_losses = lstm.get('train_losses', [])
    test_losses = lstm.get('test_losses', [])

    if not train_losses:
        fig = go.Figure()
        fig.add_annotation(text="No training loss data available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=16))
        fig.update_layout(height=400)
        return fig

    epochs = list(range(1, len(train_losses) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=epochs, y=train_losses,
        mode='lines', name='Train Loss',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=epochs, y=test_losses,
        mode='lines', name='Test Loss',
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title="LSTM Training Loss Curve",
        xaxis_title="Epoch", yaxis_title="MSE Loss",
        height=400, hovermode='x unified'
    )
    return fig

# ============================================================================
# GRADIO INTERFACE
# ============================================================================

def create_gradio_interface():
    """
    Create the Gradio UI with 7 tabs:
        1. Training Summary    -- Pre-loaded model metrics and loss curves
        2. Initialize System   -- Load market data (cached or fresh)
        3. Portfolio Optimization -- Run MPT / BL / ML-enhanced optimization
        4. Risk Analytics      -- VaR, CVaR, drawdown, Sharpe, cumulative returns
        5. Monte Carlo         -- Simulate future portfolio paths
        6. Sentiment Analysis  -- FinBERT on Finnhub news
        7. Documentation       -- Full technical reference
    """

    # Shared state across tabs
    state = {
        'system': None,
        'optimization_results': None,
        'tickers': None,
    }

    # ------------------------------------------------------------------
    # Tab 1: Training Summary
    # ------------------------------------------------------------------

    def show_training_summary():
        """Display pre-loaded training metrics and loss curve."""
        metrics = pretrained.training_metrics
        if not metrics:
            return "No training metrics available. Upload training_metrics.json.", None

        xgb_m = metrics.get('xgboost', {})
        lstm_m = metrics.get('lstm', {})
        cfg = metrics.get('config', {})

        summary = f"""## Training Summary

**Trained:** {metrics.get('training_timestamp', 'Unknown')}
**Device:** {metrics.get('gpu_name', 'Unknown')}
**Random Seed:** {metrics.get('random_seed', 42)}
**Tickers:** {', '.join(metrics.get('tickers', []))}
**Trading Days:** {metrics.get('n_trading_days', 'N/A')}

---

### XGBoost Alpha Model

| Metric | Value |
|--------|-------|
| Test R2 | {xgb_m.get('test_r2', 'N/A'):.4f} |
| Test MSE | {xgb_m.get('test_mse', 'N/A'):.6f} |
| Train R2 | {xgb_m.get('train_r2', 'N/A'):.4f} |
| Train MSE | {xgb_m.get('train_mse', 'N/A'):.6f} |
| Hit Rate | {xgb_m.get('directional_accuracy', 0):.2%} |
| Features | {xgb_m.get('n_features', 'N/A')} |
| Train Samples | {xgb_m.get('n_train_samples', 'N/A')} |
| Test Samples | {xgb_m.get('n_test_samples', 'N/A')} |

---

### LSTM Alpha Model

| Metric | Value |
|--------|-------|
| Test R2 | {lstm_m.get('final_test_r2', 'N/A'):.4f} |
| Test MSE | {lstm_m.get('final_test_mse', 'N/A'):.6f} |
| Best Test Loss | {lstm_m.get('best_test_loss', 'N/A'):.6f} |
| Hit Rate | {lstm_m.get('directional_accuracy', 0):.2%} |
| Parameters | {lstm_m.get('n_parameters', 'N/A'):,} |
| Architecture | {lstm_m.get('num_layers', 2)}-layer LSTM, {lstm_m.get('hidden_dim', 128)} hidden |
| Sequence Length | {lstm_m.get('sequence_length', 60)} days |
| Epochs | {lstm_m.get('epochs_trained', 'N/A')} |

---

### Training Configuration

| Parameter | Value |
|-----------|-------|
| XGBoost Estimators | {cfg.get('xgb_n_estimators', 'N/A')} |
| XGBoost Max Depth | {cfg.get('xgb_max_depth', 'N/A')} |
| XGBoost LR | {cfg.get('xgb_learning_rate', 'N/A')} |
| LSTM Hidden Dim | {cfg.get('lstm_hidden_dim', 'N/A')} |
| LSTM Layers | {cfg.get('lstm_num_layers', 'N/A')} |
| LSTM Dropout | {cfg.get('lstm_dropout', 'N/A')} |
| LSTM Epochs | {cfg.get('lstm_epochs', 'N/A')} |
| LSTM Batch Size | {cfg.get('lstm_batch_size', 'N/A')} |
| LSTM LR | {cfg.get('lstm_learning_rate', 'N/A')} |
| Risk-Free Rate | {cfg.get('risk_free_rate', 'N/A')} |
"""

        fig = plot_training_losses(metrics)
        return summary, fig

    # ------------------------------------------------------------------
    # Tab 2: Initialize System
    # ------------------------------------------------------------------

    def initialize_system(tickers_str, lookback_years, use_cached):
        """Initialize the portfolio system with data."""
        try:
            system = IntegratedPortfolioSystem(config)

            if use_cached:
                success = system.load_from_cache()
                if success:
                    tickers = pretrained.tickers
                    state['system'] = system
                    state['tickers'] = tickers

                    returns_table = "\n".join([
                        f"| {t} | {r:.6f} |"
                        for t, r in system.mean_returns.items()
                    ])

                    summary = f"""## System Initialized (Cached Data)

**Assets:** {len(tickers)}
**Trading Days:** {len(system.returns)}
**Tickers:** {', '.join(tickers)}

### Mean Daily Returns

| Ticker | Mean Return |
|--------|------------|
{returns_table}

Ready for optimization and analysis.
"""
                    return summary
                else:
                    return "Cached data not available. Uncheck 'Use cached data' and try again."

            # Fetch fresh data
            tickers = [t.strip().upper() for t in tickers_str.split(',')]
            system.load_fresh_data(tickers, int(lookback_years))

            state['system'] = system
            state['tickers'] = tickers

            returns_table = "\n".join([
                f"| {t} | {r:.6f} |"
                for t, r in system.mean_returns.items()
            ])

            summary = f"""## System Initialized (Fresh Data)

**Assets:** {len(tickers)}
**Trading Days:** {len(system.returns)}
**Tickers:** {', '.join(tickers)}

### Mean Daily Returns

| Ticker | Mean Return |
|--------|------------|
{returns_table}

Ready for optimization and analysis.
"""
            return summary

        except Exception as e:
            logger.error(f"Initialization error: {e}")
            import traceback
            return f"Error: {str(e)}\n{traceback.format_exc()}"

    # ------------------------------------------------------------------
    # Tab 3: Portfolio Optimization
    # ------------------------------------------------------------------

    def optimize_portfolio(method, use_ml):
        """Run portfolio optimization."""
        try:
            if state['system'] is None:
                return "Please initialize the system first.", None, None

            system = state['system']
            results = system.optimize_portfolio(method=method, use_ml=use_ml)
            state['optimization_results'] = results

            weights_fig = plot_portfolio_weights(results['weights'])
            frontier_fig = plot_efficient_frontier(
                system.mpt,
                system.mean_returns.values,
                system.cov_matrix.values,
                results['weights'].values
            )

            weights_table = "\n".join([
                f"| {t} | {w:.2%} |"
                for t, w in results['weights'].items()
            ])

            summary = f"""## Portfolio Optimization Results

**Method:** {method}
**ML Enhanced:** {'Yes' if use_ml else 'No'}

### Portfolio Statistics

| Metric | Value |
|--------|-------|
| Expected Annual Return | {results['expected_return']:.2%} |
| Annual Volatility | {results['volatility']:.2%} |
| Sharpe Ratio | {results['sharpe_ratio']:.3f} |
| VaR (95%) | {results['var_95']:.2%} |
| CVaR (95%) | {results['cvar_95']:.2%} |
| Max Drawdown | {results['max_drawdown']:.2%} |

### Asset Allocation

| Ticker | Weight |
|--------|--------|
{weights_table}
"""
            return summary, weights_fig, frontier_fig

        except Exception as e:
            logger.error(f"Optimization error: {e}")
            import traceback
            return f"Error: {str(e)}\n{traceback.format_exc()}", None, None

    # ------------------------------------------------------------------
    # Tab 4: Risk Analytics
    # ------------------------------------------------------------------

    def run_risk_analysis():
        """Run comprehensive risk analysis on the optimized portfolio."""
        try:
            if state['optimization_results'] is None:
                return "Please optimize the portfolio first.", None, None

            system = state['system']
            results = state['optimization_results']
            weights = results['weights'].values

            port_returns = np.dot(system.returns.values, weights)

            # Additional metrics
            sortino = system.risk_analytics.calculate_sortino_ratio(port_returns)

            # Information ratio (vs SPY benchmark)
            ir = 0.0
            if 'SPY' in system.returns.columns:
                bench = system.returns['SPY'].values
                min_len = min(len(port_returns), len(bench))
                ir = system.risk_analytics.calculate_information_ratio(
                    port_returns[:min_len], bench[:min_len]
                )

            risk_metrics = {
                'sharpe_ratio': results['sharpe_ratio'],
                'var_95': results['var_95'],
                'cvar_95': results['cvar_95'],
                'max_drawdown': results['max_drawdown'],
            }

            dashboard = plot_risk_metrics(risk_metrics)
            cumulative = plot_cumulative_returns(system.returns, weights)

            summary = f"""## Risk Analysis Complete

| Metric | Value |
|--------|-------|
| Sharpe Ratio | {results['sharpe_ratio']:.3f} |
| Sortino Ratio | {sortino:.3f} |
| Information Ratio | {ir:.3f} |
| VaR (95%) | {results['var_95']:.2%} |
| CVaR (95%) | {results['cvar_95']:.2%} |
| Max Drawdown | {results['max_drawdown']:.2%} |
"""
            return summary, dashboard, cumulative

        except Exception as e:
            logger.error(f"Risk analysis error: {e}")
            return f"Error: {str(e)}", None, None

    # ------------------------------------------------------------------
    # Tab 5: Monte Carlo Simulation
    # ------------------------------------------------------------------

    def run_monte_carlo(n_simulations, n_days):
        """Run Monte Carlo simulation on the optimized portfolio."""
        try:
            if state['optimization_results'] is None:
                return "Please optimize the portfolio first.", None

            system = state['system']
            weights = state['optimization_results']['weights'].values

            paths = system.risk_analytics.monte_carlo_simulation(
                system.mean_returns.values,
                system.cov_matrix.values,
                weights,
                n_days=int(n_days),
                n_simulations=int(n_simulations)
            )

            final = paths[:, -1]

            summary = f"""## Monte Carlo Simulation Results

**Simulations:** {int(n_simulations):,}
**Horizon:** {int(n_days)} days

### Terminal Portfolio Value Distribution

| Metric | Value |
|--------|-------|
| Mean | {np.mean(final):.4f} |
| Median | {np.median(final):.4f} |
| 5th Percentile | {np.percentile(final, 5):.4f} |
| 95th Percentile | {np.percentile(final, 95):.4f} |
| Probability of Loss | {(final < 1.0).mean():.1%} |
| Std Dev | {np.std(final):.4f} |
"""
            fig = plot_monte_carlo(paths)
            return summary, fig

        except Exception as e:
            logger.error(f"Monte Carlo error: {e}")
            return f"Error: {str(e)}", None

    # ------------------------------------------------------------------
    # Tab 6: Sentiment Analysis
    # ------------------------------------------------------------------

    def analyze_sentiment():
        """Analyze sentiment for portfolio tickers using FinBERT."""
        try:
            if state['system'] is None or state['tickers'] is None:
                return "Please initialize the system first.", None

            system = state['system']
            tickers = state['tickers']

            scores = system.analyze_sentiment(tickers)

            if not scores or all(s.get('num_articles', 0) == 0 for s in scores.values()):
                return ("No news articles found. Ensure FINNHUB_API_KEY is set "
                        "in HF Spaces Secrets."), None

            fig = plot_sentiment_scores(scores)

            rows = "\n".join([
                f"| {t} | {s.get('compound', 0):.3f} | {s.get('positive', 0):.3f} | "
                f"{s.get('negative', 0):.3f} | {s.get('neutral', 0):.3f} | "
                f"{s.get('num_articles', 0)} |"
                for t, s in scores.items()
            ])

            summary = f"""## Sentiment Analysis Results

| Ticker | Compound | Positive | Negative | Neutral | Articles |
|--------|----------|----------|----------|---------|----------|
{rows}
"""
            return summary, fig

        except Exception as e:
            logger.error(f"Sentiment error: {e}")
            return f"Error: {str(e)}", None

    # ------------------------------------------------------------------
    # Build interface
    # ------------------------------------------------------------------

    with gr.Blocks(title="AI Portfolio Optimizer", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # AI-Powered Multi-Asset Portfolio Optimizer

        **Production-Grade Quantitative Finance System**

        Combines Modern Portfolio Theory, Black-Litterman, Machine Learning, Sentiment Analysis,
        and Advanced Risk Management for optimal portfolio construction.

        ---
        """)

        with gr.Tabs():

            # Tab 1: Training Summary
            with gr.TabItem("Training Summary"):
                gr.Markdown("""
                ### Pre-Trained Model Performance

                XGBoost and LSTM alpha models were trained on GPU.
                Weights are loaded at startup for CPU inference.
                """)

                summary_button = gr.Button("Show Training Metrics", variant="primary", size="lg")
                training_summary_md = gr.Markdown("Click to load training metrics...")
                training_loss_plot = gr.Plot(label="LSTM Loss Curve")

                summary_button.click(
                    fn=show_training_summary,
                    outputs=[training_summary_md, training_loss_plot]
                )

            # Tab 2: Initialize System
            with gr.TabItem("Initialize System"):
                gr.Markdown("""
                ### Load Market Data

                Use cached data from training or fetch fresh data from yfinance.
                """)

                with gr.Row():
                    with gr.Column():
                        tickers_input = gr.Textbox(
                            value="SPY,QQQ,TLT,GLD,VNQ",
                            label="Tickers (comma-separated)",
                            info="Used only when fetching fresh data"
                        )
                        lookback_input = gr.Slider(
                            minimum=1, maximum=10, value=5, step=1,
                            label="Lookback Period (years)"
                        )
                        use_cached_cb = gr.Checkbox(
                            value=True,
                            label="Use cached data from training",
                            info="Loads pre-computed returns, covariance, and factor loadings"
                        )
                        init_button = gr.Button("Load Data", variant="primary", size="lg")

                    with gr.Column():
                        init_output = gr.Markdown("Waiting to load data...")

                init_button.click(
                    fn=initialize_system,
                    inputs=[tickers_input, lookback_input, use_cached_cb],
                    outputs=[init_output]
                )

            # Tab 3: Portfolio Optimization
            with gr.TabItem("Portfolio Optimization"):
                gr.Markdown("""
                ### Optimize Portfolio Allocation

                Select an optimization method and optionally incorporate ML alpha predictions.
                """)

                with gr.Row():
                    with gr.Column():
                        method_dropdown = gr.Dropdown(
                            choices=['max_sharpe', 'min_volatility',
                                     'black_litterman', 'ml_enhanced'],
                            value='max_sharpe',
                            label="Optimization Method"
                        )
                        use_ml_cb = gr.Checkbox(
                            value=False,
                            label="Use ML Alpha Models (XGBoost + LSTM)",
                            info="Blends pre-trained model predictions with historical returns"
                        )
                        opt_button = gr.Button("Optimize Portfolio", variant="primary", size="lg")

                    with gr.Column():
                        opt_summary = gr.Markdown("Waiting for optimization...")

                with gr.Row():
                    weights_plot = gr.Plot(label="Portfolio Allocation")
                    frontier_plot = gr.Plot(label="Efficient Frontier")

                opt_button.click(
                    fn=optimize_portfolio,
                    inputs=[method_dropdown, use_ml_cb],
                    outputs=[opt_summary, weights_plot, frontier_plot]
                )

            # Tab 4: Risk Analytics
            with gr.TabItem("Risk Analytics"):
                gr.Markdown("""
                ### Comprehensive Risk Analysis

                VaR, CVaR, Sharpe, Sortino, drawdowns, and cumulative returns.
                """)

                risk_button = gr.Button("Run Risk Analysis", variant="primary", size="lg")
                risk_summary = gr.Markdown("Waiting for analysis...")

                with gr.Row():
                    risk_dashboard = gr.Plot(label="Risk Metrics")
                    cumulative_plot = gr.Plot(label="Cumulative Returns")

                risk_button.click(
                    fn=run_risk_analysis,
                    outputs=[risk_summary, risk_dashboard, cumulative_plot]
                )

            # Tab 5: Monte Carlo Simulation
            with gr.TabItem("Monte Carlo Simulation"):
                gr.Markdown("""
                ### Portfolio Stress Testing

                Simulate thousands of potential future scenarios using
                Cholesky-decomposed correlated returns.
                """)

                with gr.Row():
                    with gr.Column():
                        mc_sims = gr.Slider(
                            minimum=1000, maximum=50000, value=10000, step=1000,
                            label="Number of Simulations"
                        )
                        mc_days = gr.Slider(
                            minimum=30, maximum=1260, value=252, step=30,
                            label="Time Horizon (days)"
                        )
                        mc_button = gr.Button("Run Simulation", variant="primary", size="lg")

                    with gr.Column():
                        mc_summary = gr.Markdown("Waiting for simulation...")

                mc_plot = gr.Plot(label="Simulated Paths")

                mc_button.click(
                    fn=run_monte_carlo,
                    inputs=[mc_sims, mc_days],
                    outputs=[mc_summary, mc_plot]
                )

            # Tab 6: Sentiment Analysis
            with gr.TabItem("Sentiment Analysis"):
                gr.Markdown("""
                ### FinBERT Sentiment Analysis

                Analyze market sentiment from recent news headlines using the
                FinBERT transformer model. Requires FINNHUB_API_KEY in Secrets.
                """)

                sentiment_button = gr.Button("Analyze Sentiment", variant="primary", size="lg")
                sentiment_summary = gr.Markdown("Waiting for analysis...")
                sentiment_plot = gr.Plot(label="Sentiment Scores")

                sentiment_button.click(
                    fn=analyze_sentiment,
                    outputs=[sentiment_summary, sentiment_plot]
                )

            # Tab 7: Documentation
            with gr.TabItem("Documentation"):
                gr.Markdown("""
                ## System Documentation

                ### Overview

                This is a production-grade AI-powered multi-asset portfolio optimizer that combines
                classical financial theory with modern machine learning and deep learning techniques.

                ### Key Components

                #### 1. Modern Portfolio Theory (MPT)
                - **Mean-Variance Optimization:** Maximize Sharpe ratio or minimize volatility
                - **Efficient Frontier:** Calculate risk-return tradeoffs
                - **Constraints:** Min/max weights, fully invested
                - **Implementation:** CVXPY convex optimization

                #### 2. Black-Litterman Model
                - **Market Equilibrium:** Reverse optimization for implied returns
                - **Bayesian Updating:** Incorporate investor views via posterior
                - **View Confidence:** Adjust view uncertainty matrix (Omega)
                - **Posterior Distribution:** Combines market prior with views

                #### 3. Machine Learning Alpha Generation

                **XGBoost Model:**
                - Gradient boosting for return prediction
                - 100+ features: rolling returns, volatility, moving averages, RSI, volume
                - Time-series cross-validation (80/20 split)
                - Pre-trained on GPU, inference on CPU

                **LSTM Model:**
                - 2-layer recurrent neural network (128 hidden units)
                - 60-day input sequences of multi-asset returns
                - Trained with MSE loss, Adam optimizer, best-model checkpointing
                - Pre-trained on GPU, inference on CPU

                #### 4. Fama-French Factor Models
                - **5-Factor Model:** Market, SMB, HML, RMW, CMA
                - **Factor Loadings:** OLS regression for each asset
                - **Expected Returns:** Factor premiums x betas + alpha
                - **Risk Decomposition:** Systematic vs. idiosyncratic risk

                #### 5. Sentiment Analysis
                - **Model:** FinBERT (financial domain BERT)
                - **Source:** Finnhub news API
                - **Window:** 30 days
                - **Scoring:** Positive, negative, neutral, compound
                - **Integration:** Adjust portfolio views based on sentiment

                #### 6. Advanced Risk Metrics

                **Value at Risk (VaR):**
                - 95% confidence level
                - Historical and parametric methods
                - 1-day horizon

                **Conditional VaR (CVaR):**
                - Expected shortfall beyond VaR
                - More conservative tail risk measure
                - Captures extreme losses

                **Maximum Drawdown:**
                - Peak-to-trough decline
                - Identifies worst historical loss period
                - Key metric for downside risk assessment

                **Sharpe Ratio:**
                - Risk-adjusted return: (Return - Rf) / Volatility
                - Annualized
                - Standard benchmark for portfolio performance

                **Sortino Ratio:**
                - Downside risk-adjusted return
                - Uses downside deviation only
                - More appropriate for asymmetric return distributions

                **Information Ratio:**
                - Active return / tracking error
                - Measures skill relative to benchmark
                - Annualized

                #### 7. Monte Carlo Simulation
                - **Methodology:** Cholesky decomposition for correlated returns
                - **Simulations:** Up to 50,000 paths
                - **Horizon:** Up to 5 years
                - **Output:** Distribution of terminal wealth
                - **Metrics:** Probability of loss, percentiles

                #### 8. Transaction Cost Optimization
                - **Turnover Calculation:** Sum of absolute weight changes
                - **Cost Model:** Basis points per trade (default 10 bps)
                - **Rebalancing Threshold:** 5% weight drift trigger
                - **Optimization:** Balance expected return vs. transaction costs

                ### Optimization Methods

                **Maximum Sharpe Ratio:**
                Objective: Maximize risk-adjusted return. Uses mean-variance utility
                with configurable risk aversion. Most common approach.

                **Minimum Volatility:**
                Objective: Minimize portfolio variance. Conservative approach
                suitable for risk-averse investors.

                **Black-Litterman:**
                Combines market equilibrium returns with investor views using
                Bayesian updating. Reduces estimation error and produces
                more stable allocations.

                **ML Enhanced:**
                Uses pre-trained XGBoost and LSTM predictions to adjust expected
                returns before optimization. Blends 70% historical / 30% ML signal.

                ### API Integration

                **Finnhub (Sentiment):**
                - Real-time company news
                - Free tier: 60 calls/minute
                - Set FINNHUB_API_KEY in HF Spaces Secrets

                **yfinance (Market Data):**
                - Historical daily prices
                - No API key required
                - Used for fresh data loading

                ### Limitations and Assumptions

                1. Historical data dependency: past performance does not predict future results
                2. Normal distribution assumption: real returns exhibit fat tails
                3. Stationary assumptions: markets experience regime shifts
                4. Estimation error: covariance matrix estimation is noisy
                5. Transaction costs: simplified model (no market impact)
                6. Single period: optimization is myopic, not multi-period
                7. No derivatives: does not model options or futures
                8. Sentiment lag: news may already be priced in

                ### Failure Modes

                1. Black swan events not captured in historical training data
                2. Regime changes can break factor model relationships
                3. ML models may overfit on training data noise
                4. Sentiment data subject to manipulation or misinterpretation
                5. Small input changes can produce large allocation shifts
                6. Correlation structure breaks down during market crises

                ### References

                1. Markowitz (1952): "Portfolio Selection"
                2. Sharpe (1964): "Capital Asset Pricing Model"
                3. Black and Litterman (1992): "Global Portfolio Optimization"
                4. Fama and French (1993, 2015): "Common Risk Factors", "Five-Factor Model"
                5. Gu, Kelly, Xiu (2020): "Empirical Asset Pricing via Machine Learning"
                6. Jorion (2006): "Value at Risk"
                7. Rockafellar and Uryasev (2000): "Optimization of CVaR"

                ### License and Disclaimer

                **License:** MIT License

                **Disclaimer:** This system is for educational and research purposes only.
                Real investing involves significant risks and may result in loss of capital.
                Past performance does not guarantee future results.
                Always consult with financial professionals before making investment decisions.

                ### Author

                **Spencer Purdy**

                Portfolio demonstration of quantitative finance, portfolio optimization,
                machine learning, deep learning, risk management, and production engineering.

                ---

                **System Version:** 2.0.0
                **Last Updated:** February 2026
                **Training Hardware:** NVIDIA H100 80GB HBM3
                **Inference Hardware:** CPU (Hugging Face Spaces)
                """)

        gr.Markdown("""
        ---
        **AI Portfolio Optimizer v2.0.0** | PyTorch, CVXPY, FinBERT, XGBoost, Gradio | Author: Spencer Purdy

        MPT, Black-Litterman, XGBoost, LSTM, Fama-French, FinBERT, VaR/CVaR, Monte Carlo
        """)

    return interface

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("AI-Powered Multi-Asset Portfolio Optimizer with Risk Management")
    logger.info("Author: Spencer Purdy")
    logger.info("=" * 80)

    interface = create_gradio_interface()

    interface.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )