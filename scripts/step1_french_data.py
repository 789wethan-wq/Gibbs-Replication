"""00_data_pipeline.py — Download and clean all data from the Ken French library.

Portfolio-level test path: no yfinance required. Market temperature T is built
from the FF *daily* market return (Mkt-RF + RF), avoiding any external price feed.
"""
import io, os, zipfile
import urllib.request
import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.abspath(__file__)))

BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
DATA = "../data"
os.makedirs(DATA, exist_ok=True)

FILES = {
    "ff5_m": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "ff5_d": "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "ff25_m": "25_Portfolios_5x5_CSV.zip",
    "mom_m": "F-F_Momentum_Factor_CSV.zip",
}


def fetch_zip_csv(fname):
    url = BASE + fname
    raw = urllib.request.urlopen(url, timeout=60).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    return z.read(name).decode("latin-1")


def parse_french(text, monthly=True):
    """Parse a French CSV: returns the FIRST data block (rows whose first token
    is an 6- or 8-digit date). Returns DataFrame indexed by period-end timestamp."""
    lines = text.splitlines()
    rows, header = [], None
    datelen = 6 if monthly else 8
    for ln in lines:
        parts = [p.strip() for p in ln.split(",")]
        tok = parts[0]
        if header is None:
            # header row: blank first cell, rest are column names
            if tok == "" and any(parts[1:]):
                header = [c for c in parts[1:] if c != ""]
            continue
        if tok.isdigit() and len(tok) == datelen:
            vals = parts[1 : 1 + len(header)]
            try:
                fvals = [float(v) for v in vals]
            except ValueError:
                continue
            if len(fvals) == len(header):
                rows.append([tok] + fvals)
        elif rows:
            # reached the end of the first numeric block (annual section / notes)
            break
    df = pd.DataFrame(rows, columns=["date"] + header)
    if monthly:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    else:
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    return df.set_index("date")


def main():
    print("Downloading Ken French data library files...")
    texts = {k: fetch_zip_csv(v) for k, v in FILES.items()}

    # --- FF5 monthly factors (in percent) ---
    ff5 = parse_french(texts["ff5_m"], monthly=True) / 100.0
    ff5.columns = [c.replace("-", "_") for c in ff5.columns]  # Mkt_RF, SMB, HML, RMW, CMA, RF
    ff5.to_parquet(f"{DATA}/ff5_monthly.parquet")

    # --- Momentum monthly ---
    mom = parse_french(texts["mom_m"], monthly=True) / 100.0
    mom.columns = ["Mom"]
    factors = ff5.join(mom, how="left")
    factors["Mom"] = factors["Mom"].fillna(0.0)
    factors.to_parquet(f"{DATA}/factors_monthly.parquet")

    # --- 25 portfolios (Size x B/M), value-weighted monthly, in percent ---
    ff25 = parse_french(texts["ff25_m"], monthly=True) / 100.0
    ff25.to_parquet(f"{DATA}/ff25_returns.parquet")

    # --- Daily FF5 -> market return -> temperature T ---
    ff5d = parse_french(texts["ff5_d"], monthly=False) / 100.0
    ff5d.columns = [c.replace("-", "_") for c in ff5d.columns]
    mkt_d = ff5d["Mkt_RF"] + ff5d["RF"]  # total daily market return
    logret = np.log1p(mkt_d.clip(lower=-0.99))
    sp500_daily = pd.DataFrame({"mkt_ret": mkt_d, "log_ret": logret})
    sp500_daily.to_parquet(f"{DATA}/sp500_daily.parquet")

    # 12-month (252 trading day) realized variance, annualized
    rv = logret.pow(2).rolling(252).sum() * 1.0  # sum of sq daily log returns over ~1yr
    # annualize: already ~252 obs summed => annual variance
    T_monthly = rv.resample("ME").last().dropna()
    T_monthly.name = "T_raw"
    T_monthly.to_frame().to_parquet(f"{DATA}/market_temperature.parquet")

    # --- Data quality / coverage report ---
    print("\n=== DATA PIPELINE REPORT ===")
    print(f"FF5 monthly:   {ff5.index.min():%Y-%m} to {ff5.index.max():%Y-%m}  ({len(ff5)} months)")
    print(f"FF25 monthly:  {ff25.index.min():%Y-%m} to {ff25.index.max():%Y-%m}  ({ff25.shape[1]} portfolios)")
    print(f"FF5 daily:     {ff5d.index.min():%Y-%m-%d} to {ff5d.index.max():%Y-%m-%d}  ({len(ff5d)} days)")
    print(f"Temperature T: {T_monthly.index.min():%Y-%m} to {T_monthly.index.max():%Y-%m}")
    print(f"  T raw  mean={T_monthly.mean():.4f}  std={T_monthly.std():.4f}  "
          f"min={T_monthly.min():.4f}  max={T_monthly.max():.4f}")
    print(f"Missing FF25 cells: {int(ff25.isna().sum().sum())}")
    print(f"Data pipeline complete. {ff25.shape[1]} portfolios, {len(ff25)} months, "
          f"range {ff25.index.min():%Y-%m} to {ff25.index.max():%Y-%m}")


if __name__ == "__main__":
    main()
