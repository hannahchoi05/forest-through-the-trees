import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

def check_parity(py_path, r_path, name):
    print(f"--- Checking Parity for {name} ---")
    try:
        df_py = pd.read_csv(py_path)
        df_r = pd.read_csv(r_path)
        
        if df_py.shape != df_r.shape:
            print(f"Shape mismatch: Py {df_py.shape} vs R {df_r.shape}")
            return
            
        py_vals = df_py.values
        r_vals = df_r.values
        
        diff = np.abs(py_vals - r_vals)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        print(f"Max absolute diff: {max_diff}")
        print(f"Mean absolute diff: {mean_diff}")
        if max_diff < 1e-10:
            print("Parity: PERFECT MATCH")
        else:
            print("Parity: MISMATCH DETECTED")
    except Exception as e:
        print(f"Error checking {name}: {e}")

if __name__ == '__main__':
    # Check TS32
    py_ts32 = os.path.join(base_dir, "ts_portfolio_py/LME_OP_Investment/excess_ports.csv")
    r_ts32 = os.path.join(base_dir, "../Data/ts_portfolio/LME_OP_Investment/excess_ports.csv")
    check_parity(py_ts32, r_ts32, "TS32")
    
    # Check TS64
    py_ts64 = os.path.join(base_dir, "ts64_portfolio_py/LME_OP_Investment/excess_ports.csv")
    r_ts64 = os.path.join(base_dir, "../Data/ts64_portfolio/LME_OP_Investment/excess_ports.csv")
    check_parity(py_ts64, r_ts64, "TS64")

