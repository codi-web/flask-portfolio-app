import os
import base64
from io import BytesIO
from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
from portfolio_utils import (
    download_data_with_retry, verify_symbol,
    sortino_ratio, sharpe_ratio, calculate_cvar,
    monte_carlo_simulation, optimize_portfolio,
    generate_graph_base64
)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    graph_types = [
        'normalized_prices', 'cumulative_returns', 'returns_distribution',
        'annual_volatility', 'rolling_volatility', 'correlation_matrix',
        'efficient_frontier', 'optimal_allocation_pct', 'optimal_allocation_eur',
        'max_return_allocation', 'compare_benchmark', 'beta_sensitivity',
        'scenario_analysis'
    ]
    if request.method == 'POST':
        symbols = [s.strip().upper() for s in request.form['symbols'].split(',') if s.strip()]
        start = request.form['start_date']
        end = request.form['end_date']
        sim_count = int(request.form['simulations'])
        invest_amount = float(request.form['investment'])
        rf_rate = float(request.form['risk_free']) / 100
        selected_graph = request.form['graph_type']

        symbols = [s for s in symbols if verify_symbol(s)]
        prices = download_data_with_retry(symbols, start, end)
        log_returns = np.log(1 + prices.pct_change()).dropna()
        cov_matrix = log_returns.cov() * 252
        num_assets = len(symbols)

        mc_ret, mc_vol, mc_shp, mc_w = monte_carlo_simulation(
            log_returns, cov_matrix, sim_count, num_assets, rf_rate
        )
        idx_opt = np.argmax(mc_shp)
        weights_opt = mc_w[idx_opt]
        metrics = {
            'ret_opt': mc_ret[idx_opt],
            'vol_opt': mc_vol[idx_opt],
            'shp_opt': mc_shp[idx_opt],
            'sortino_opt': sortino_ratio((log_returns * weights_opt).sum(axis=1), rf_rate),
            'cvar_opt': calculate_cvar((log_returns * weights_opt).sum(axis=1))
        }

        graph_img = generate_graph_base64(
            selected_graph, prices, log_returns, sim_count,
            mc_ret, mc_vol, mc_shp, mc_w, weights_opt,
            metrics, invest_amount, request.form.get('benchmark','')
        )

        return render_template('results.html',
                               symbols=symbols,
                               metrics=metrics,
                               graph_img=graph_img,
                               graph_name=selected_graph,
                               graph_types=graph_types)
    return render_template('index.html', graph_types=graph_types)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 80))
    app.run(host='0.0.0.0', port=port)
