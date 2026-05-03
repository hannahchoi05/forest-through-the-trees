import os
import numpy as np
import pandas as pd

def ntile_r(x, n):
    """Mimic dplyr::ntile behavior to match the R results exactly."""
    x_rank = x.rank(method='first').astype(int) - 1
    size = len(x)
    base_size = size // n
    remainder = size % n
    
    buckets = np.where(
        x_rank < remainder * (base_size + 1),
        x_rank // (base_size + 1),
        remainder + (x_rank - remainder * (base_size + 1)) // base_size
    )
    
    return buckets + 1

def triple_sort_helper(df_tmp, feat1, feat2, feat3, n_bins=(4, 4, 4)):
    n_ports = n_bins[0] * n_bins[1] * n_bins[2]
    ret_tmp = np.zeros((n_ports, 12))
    
    for mon in range(1, 13):
        mask_m = df_tmp['mm'] == mon
        df_m = df_tmp[mask_m].copy()
        
        if df_m.empty:
            continue
            
        df_m['1'] = ntile_r(df_m[feat1], n_bins[0])
        df_m['2'] = ntile_r(df_m[feat2], n_bins[1])
        df_m['3'] = ntile_r(df_m[feat3], n_bins[2])
        
        for i in range(1, n_bins[0] + 1):
            for j in range(1, n_bins[1] + 1):
                for k in range(1, n_bins[2] + 1):
                    mask_port = (df_m['1'] == i) & (df_m['2'] == j) & (df_m['3'] == k)
                    df_port = df_m[mask_port]
                    
                    company_val = df_port['size'].values
                    ret_mon = df_port['ret'].values
                    
                    idx = (i-1) * (n_bins[1] * n_bins[2]) + (j-1) * n_bins[2] + (k-1)
                    if np.sum(company_val) != 0:
                        ret_tmp[idx, mon-1] = np.sum(ret_mon * company_val) / np.sum(company_val)
                    else:
                        ret_tmp[idx, mon-1] = np.nan
    return ret_tmp

def triple_sort(data_path, feat1, feat2, feat3, y_min, y_max, n_bins=(4, 4, 4)):
    n_ports = n_bins[0] * n_bins[1] * n_bins[2]
    num_years = y_max - y_min + 1
    ret_table = np.zeros((n_ports, num_years * 12))
    
    y_time_stamp = 1
    for y in range(y_min, y_max + 1):
        print(f"Processing year {y}...")
        data_filenm = os.path.join(data_path, f"y{y}.csv")
        df_tmp = pd.read_csv(data_filenm)
        
        ret_tmp = triple_sort_helper(df_tmp, feat1, feat2, feat3, n_bins)
        ret_table[:, (y_time_stamp-1)*12 : y_time_stamp*12] = ret_tmp
        y_time_stamp += 1
        
    return ret_table

def remove_rf(port_ret, factor_path):
    file_nm = os.path.join(factor_path, 'rf_factor.csv')
    r_f = pd.read_csv(file_nm, header=None).values.flatten()
    
    port_ret_adjusted = port_ret.copy()
    for i in range(port_ret.shape[1]):
        port_ret_adjusted[:, i] = port_ret[:, i] - (r_f / 100)
        
    return port_ret_adjusted

def genTripleSort(feats_list, feat1_idx, feat2_idx, y_min, y_max, data_chunk_path, output_path, factor_path, n_bins=(4, 4, 4)):
    print("feat1:", feat1_idx)
    print("feat2:", feat2_idx)
    feats = ['LME', feats_list[feat1_idx], feats_list[feat2_idx]]
    
    sub_dir = f"{feats[0]}_{feats[1]}_{feats[2]}"
    os.makedirs(os.path.join(output_path, sub_dir), exist_ok=True)
    data_path = os.path.join(data_chunk_path, sub_dir) + '/'
    
    ret_table = triple_sort(data_path, feats[0], feats[1], feats[2], y_min, y_max, n_bins)
    print("NA count before transposition:", np.sum(np.isnan(ret_table)))
    
    ret_table = ret_table.T
    port_ret = remove_rf(ret_table, factor_path)
    port_ret = np.nan_to_num(port_ret, nan=0.0)
    
    output_file = os.path.join(output_path, sub_dir, 'excess_ports.csv')
    df_out = pd.DataFrame(port_ret)
    df_out.columns = [f"V{i+1}" for i in range(port_ret.shape[1])]
    df_out.to_csv(output_file, index=False)
    
    print(f"Output written to {output_file}. Shape: {port_ret.shape}")

def genTripleSort64(feats_list, feat1_idx, feat2_idx, y_min, y_max, data_chunk_path, output_path, factor_path):
    genTripleSort(feats_list, feat1_idx, feat2_idx, y_min, y_max, data_chunk_path, output_path, factor_path, n_bins=(4, 4, 4))

def genTripleSort32(feats_list, feat1_idx, feat2_idx, y_min, y_max, data_chunk_path, output_path, factor_path):
    genTripleSort(feats_list, feat1_idx, feat2_idx, y_min, y_max, data_chunk_path, output_path, factor_path, n_bins=(2, 4, 4))

if __name__ == "__main__":
    feats_list = ['LME','BEME','r12_2','OP','Investment','ST_Rev','LT_Rev','AC','IdioVol',"LTurnover"]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    genTripleSort64(
        feats_list=feats_list,
        feat1_idx=3,
        feat2_idx=4,
        y_min=1964,
        y_max=2016,
        data_chunk_path=os.path.join(base_dir, "../Data/data_chunk_files_quantile/"),
        output_path=os.path.join(base_dir, "ts64_portfolio_py/"),
        factor_path=os.path.join(base_dir, "../Data/factor/")
    )
    
    genTripleSort32(
        feats_list=feats_list,
        feat1_idx=3,
        feat2_idx=4,
        y_min=1964,
        y_max=2016,
        data_chunk_path=os.path.join(base_dir, "../Data/data_chunk_files_quantile/"),
        output_path=os.path.join(base_dir, "ts_portfolio_py/"),
        factor_path=os.path.join(base_dir, "../Data/factor/")
    )
