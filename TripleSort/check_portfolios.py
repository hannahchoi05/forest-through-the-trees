from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _abs_paths() -> tuple[Path, Path]:
    base = Path(__file__).resolve().parent
    repo = base.parent
    return base, repo


def check_parity(py_path: Path, data_path: Path, name: str, tol: float = 1e-10) -> None:
    df_py = pd.read_csv(py_path)
    df_data = pd.read_csv(data_path)

    if df_py.shape != df_data.shape:
        raise AssertionError(f"{name}: shape mismatch: py {df_py.shape} vs data {df_data.shape}")

    diff = np.abs(df_py.values - df_data.values)
    max_diff = float(np.nanmax(diff))
    mean_diff = float(np.nanmean(diff))

    if not (max_diff <= tol):
        raise AssertionError(f"{name}: mismatch: max_diff={max_diff} mean_diff={mean_diff}")


def check_all() -> None:
    base, repo = _abs_paths()

    py_ts32 = base / "ts_portfolio_py" / "LME_OP_Investment" / "excess_ports.csv"
    data_ts32 = repo / "Data" / "ts_portfolio" / "LME_OP_Investment" / "excess_ports.csv"
    check_parity(py_ts32, data_ts32, "TS32")

    py_ts64 = base / "ts64_portfolio_py" / "LME_OP_Investment" / "excess_ports.csv"
    data_ts64 = repo / "Data" / "ts64_portfolio" / "LME_OP_Investment" / "excess_ports.csv"
    check_parity(py_ts64, data_ts64, "TS64")


if __name__ == "__main__":
    check_all()
    print("TripleSort portfolio parity checks: OK")
