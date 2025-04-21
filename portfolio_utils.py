import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import seaborn as sns

def download_data_with_retry(symbols, start_date, end_date, retries=3):
    for i in range(retries):
        try:
            data = yf.download(symbols, start=start_date, end=end_date, progress=False)
            if isinstance(data.columns, pd.MultiIndex):
                prices = data['Close']
            else:
                prices = data[['Close']]
                prices.columns = [symbols[0]]
            return prices.ffill().bfill()
        except Exception:
            continue
    raise RuntimeError('Error descargando datos')

def verify_symbol(symbol):
    try:
        return not yf.Ticker(symbol).history(period='5d').empty
    except:
        return False

def sortino_ratio(returns, risk_free_rate=0.0):
    neg = returns[returns < 0]
    downside = neg.std()
    if downside == 0:
        return 0.0
    ann_ret = returns.mean() * 252
    return (ann_ret - risk_free_rate) / (downside * np.sqrt(252))

def sharpe_ratio(returns, risk_free_rate=0.0):
    vol = returns.std()
    if vol == 0:
        return 0.0
    ann_ret = returns.mean() * 252
    return (ann_ret - risk_free_rate) / (vol * np.sqrt(252))

def calculate_cvar(returns, level=0.95):
    var = np.percentile(returns, (1 - level) * 100)
    tail = returns[returns <= var]
    return tail.mean() if not tail.empty else var

def monte_carlo_simulation(log_returns, cov_matrix, n_portfolios, n_assets, rf_rate):
    results = np.zeros((3, n_portfolios))
    weights = []
    mean_ret = log_returns.mean() * 252
    for i in range(n_portfolios):
        w = np.random.random(n_assets)
        w /= w.sum()
        pr = np.dot(w, mean_ret)
        pv = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        sr = (pr - rf_rate) / pv if pv > 0 else 0
        results[:, i] = [pr, pv, sr]
        weights.append(w)
    return results[0], results[1], results[2], weights

def optimize_portfolio(log_returns, cov_matrix, n_assets):
    def vol_fn(w):
        return np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    bnds = tuple((0, 1) for _ in range(n_assets))
    init = np.repeat(1 / n_assets, n_assets)
    res = minimize(vol_fn, init, method='SLSQP', bounds=bnds, constraints=cons)
    return res.x if res.success else init

def generate_graph_base64(graph_type, prices, log_returns, n_sim,
                          mc_ret, mc_vol, mc_shp, mc_w,
                          weights_opt, metrics, invest_amount, benchmark):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.set_theme(style='whitegrid')
    assets = prices.columns.tolist()

    if graph_type == 'normalized_prices':
        norm = prices / prices.iloc[0] * 100
        norm.plot(ax=ax)
        ax.set_title('Precios Normalizados (Base = 100)')
    # ... rest of graph generation logic ...
    plt.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()
