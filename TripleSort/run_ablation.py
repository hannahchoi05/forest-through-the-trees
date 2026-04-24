import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib.pyplot as plt
import os

def solve_portfolio(mu, Sigma, lambda1, lambda2, mu0, w_prev=None, lambda_tc=0.0):
    n = len(mu)
    w = cp.Variable(n)
    
    variance = cp.quad_form(w, Sigma)
    l1_penalty = cp.norm1(w)
    l2_penalty = cp.sum_squares(w)
    
    objective = 0.5 * variance + lambda1 * l1_penalty + 0.5 * lambda2 * l2_penalty
    
    if lambda_tc > 0 and w_prev is not None:
        tc_penalty = lambda_tc * cp.norm1(w - w_prev)
        objective += tc_penalty
        
    constraints = [
        cp.sum(w) == 1,
        w @ mu >= mu0
    ]
    
    prob = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        prob.solve(solver=cp.OSQP, max_iter=10000, eps_abs=1e-5, eps_rel=1e-5)
        if prob.status not in ["optimal", "optimal_inaccurate"]:
            prob.solve(solver=cp.ECOS)
        return w.value
    except Exception as e:
        print("Solver failed:", e)
        return None

def compute_metrics(rgross, rnet, to, c):
    T = len(rgross)
    mean_g = np.mean(rgross)
    mean_n = np.mean(rnet)
    
    mu_ann_g = 12 * mean_g
    mu_ann_n = 12 * mean_n
    
    vol_g = np.std(rgross)
    vol_n = np.std(rnet)
    
    vol_ann_g = np.sqrt(12) * vol_g
    vol_ann_n = np.sqrt(12) * vol_n
    
    sr_g = mean_g / vol_g if vol_g > 0 else 0
    sr_n = mean_n / vol_n if vol_n > 0 else 0
    
    sr_ann_g = np.sqrt(12) * sr_g
    sr_ann_n = np.sqrt(12) * sr_n
    
    w_n = np.cumprod(1 + rnet)
    peak_n = np.maximum.accumulate(w_n)
    dd_n = (w_n / peak_n) - 1
    mdd_n = np.min(dd_n)
    
    hit_g = np.mean(rgross > 0)
    hit_n = np.mean(rnet > 0)
    
    avg_to = np.mean(to)
    avg_cost = np.mean(to * c)
    
    delta_sr = sr_ann_g - sr_ann_n
    
    return {
        'mu_ann_g': mu_ann_g, 'mu_ann_n': mu_ann_n,
        'vol_ann_g': vol_ann_g, 'vol_ann_n': vol_ann_n,
        'sr_ann_g': sr_ann_g, 'sr_ann_n': sr_ann_n,
        'mdd_n': mdd_n,
        'hit_g': hit_g, 'hit_n': hit_n,
        'avg_to': avg_to, 'avg_cost': avg_cost,
        'delta_sr': delta_sr
    }

def run_ablation():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    port_ret = pd.read_csv(os.path.join(base_dir, 'ts_portfolio_py/LME_OP_Investment/excess_ports.csv')).values
    n_ports = port_ret.shape[1]
    factor_mat = pd.read_csv(os.path.join(base_dir, '../Data/factor/tradable_factors.csv'))
    sp500_ret = factor_mat['Mkt-RF'].values + (pd.read_csv(os.path.join(base_dir, '../Data/factor/rf_factor.csv'), header=None)[0].values / 100)
    
    L = 120
    t_start = 360 # Test sample from month 360 to 635 (inclusive)
    t_end = 636
    
    lambda1 = 0.001
    lambda2 = 0.01
    mu0 = 0.01
    lambda_tc = 0.0025
    c = 0.0025
    
    w_base_list = []
    w_tc_list = []
    
    prev_w_base = np.ones(n_ports) / n_ports
    prev_w_tc = np.ones(n_ports) / n_ports
    
    print("Running optimization...")
    # We need w_t for t = 358 to 634.
    for t in range(t_start - 2, t_end - 1):
        train_data = port_ret[t-L:t, :]
        mu_hat = np.mean(train_data, axis=0)
        Sigma_hat = np.cov(train_data, rowvar=False) + np.eye(n_ports)*1e-6
        
        wb = solve_portfolio(mu_hat, Sigma_hat, lambda1, lambda2, mu0)
        if wb is None: wb = prev_w_base
            
        wt = solve_portfolio(mu_hat, Sigma_hat, lambda1, lambda2, mu0, w_prev=prev_w_tc, lambda_tc=lambda_tc)
        if wt is None: wt = prev_w_tc
            
        w_base_list.append(wb)
        w_tc_list.append(wt)
        
        prev_w_base = wb
        prev_w_tc = wt

    w_base_arr = np.array(w_base_list)
    w_tc_arr = np.array(w_tc_list)
    
    test_ret = port_ret[t_start:t_end, :]
    sp500_test = sp500_ret[t_start:t_end]
    
    # Calculate returns
    w_base_eval = w_base_arr[1:] # w_t for t = 359 to 634
    w_tc_eval = w_tc_arr[1:]
    
    rgross_base = np.sum(w_base_eval * test_ret, axis=1)
    to_base = np.sum(np.abs(w_base_arr[1:] - w_base_arr[:-1]), axis=1)
    cost_base = c * to_base
    rnet_base = rgross_base - cost_base
    
    rgross_tc = np.sum(w_tc_eval * test_ret, axis=1)
    to_tc = np.sum(np.abs(w_tc_arr[1:] - w_tc_arr[:-1]), axis=1)
    cost_tc = c * to_tc
    rnet_tc = rgross_tc - cost_tc
    
    metrics_base = compute_metrics(rgross_base, rnet_base, to_base, c)
    metrics_tc = compute_metrics(rgross_tc, rnet_tc, to_tc, c)
    
    print("Baseline Metrics:", metrics_base)
    print("TC Metrics:", metrics_tc)
    
    # Write metrics
    metrics_path = os.path.join(base_dir, 'metrics.csv')
    with open(metrics_path, 'w') as f:
        f.write("Metric,Baseline,TransactionCost\n")
        for k in metrics_base.keys():
            f.write(f"{k},{metrics_base[k]},{metrics_tc[k]}\n")
            
    # Plots
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot A: Cumulative Gross Returns
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumprod(1 + rgross_base), label='Baseline Gross')
    plt.plot(np.cumprod(1 + rgross_tc), label='TC Gross')
    plt.plot(np.cumprod(1 + sp500_test), label='S&P 500', color='black', linestyle='--')
    plt.title('Plot A: Cumulative Gross Returns')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'plot_A.png'))
    plt.close()
    
    # Plot B: Cumulative Net Returns
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumprod(1 + rnet_base), label='Baseline Net')
    plt.plot(np.cumprod(1 + rnet_tc), label='TC Net')
    plt.title('Plot B: Cumulative Net Returns')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'plot_B.png'))
    plt.close()
    
    # Plot C: Gross vs Net Returns
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    axes[0].plot(np.cumprod(1 + rgross_base), label='Baseline Gross')
    axes[0].plot(np.cumprod(1 + rnet_base), label='Baseline Net')
    axes[0].set_title('Baseline Gross vs Net')
    axes[0].legend()
    
    axes[1].plot(np.cumprod(1 + rgross_tc), label='TC Gross')
    axes[1].plot(np.cumprod(1 + rnet_tc), label='TC Net')
    axes[1].set_title('TC Gross vs Net')
    axes[1].legend()
    plt.savefig(os.path.join(plots_dir, 'plot_C.png'))
    plt.close()
    
    # Plot D: Drawdown
    plt.figure(figsize=(10, 6))
    w_n_b = np.cumprod(1 + rnet_base)
    dd_b = w_n_b / np.maximum.accumulate(w_n_b) - 1
    w_n_t = np.cumprod(1 + rnet_tc)
    dd_t = w_n_t / np.maximum.accumulate(w_n_t) - 1
    plt.plot(dd_b, label='Baseline Net DD')
    plt.plot(dd_t, label='TC Net DD')
    plt.title('Plot D: Drawdown')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'plot_D.png'))
    plt.close()
    
    # Plot E: Turnover
    plt.figure(figsize=(10, 6))
    plt.plot(pd.Series(to_base).rolling(12).mean(), label='Baseline Turnover (12m MA)')
    plt.plot(pd.Series(to_tc).rolling(12).mean(), label='TC Turnover (12m MA)')
    plt.title('Plot E: Turnover')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'plot_E.png'))
    plt.close()
    
    # Plot F: Summary Metrics
    labels = ['Triple sort baseline', 'Triple sort TC']
    metrics_to_plot = ['sr_ann_g', 'sr_ann_n', 'avg_to', 'mdd_n']
    
    x = np.arange(len(labels))
    width = 0.2
    
    plt.figure(figsize=(10, 6))
    for i, m in enumerate(metrics_to_plot):
        vals = [metrics_base[m], metrics_tc[m]]
        offset = (i - 1.5) * width
        plt.bar(x + offset, vals, width, label=m)
        
    plt.xticks(x, labels)
    plt.title('Plot F: Summary Metrics')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'plot_F.png'))
    plt.close()

if __name__ == '__main__':
    run_ablation()
