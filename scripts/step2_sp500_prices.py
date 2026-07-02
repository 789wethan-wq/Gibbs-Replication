"""00b_stock_data.py — Download S&P 500 monthly prices via yfinance.

Uses price-based proxies for ΔH and ΔS (same as the portfolio-level fallback,
but now at stock level with ~400-500 names per cross-section). Fundamentals
from yfinance only go back ~4 years, so ROE/EPS-based proxies are reserved
for a future robustness table noted in the paper.

Survivorship bias: universe = current S&P 500 constituents. Delisted firms
excluded. Flags this in all output.
"""
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import os

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
DATA = "../data"
os.makedirs(DATA, exist_ok=True)

START = "1988-01-01"   # extra buffer for rolling windows
END   = "2023-12-31"


def get_sp500_tickers():
    # Try Wikipedia with a browser-like user-agent
    import urllib.request
    try:
        req = urllib.request.Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read()
        tables = pd.read_html(html)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
        print(f"  Got {len(tickers)} tickers from Wikipedia")
        return tickers
    except Exception as e:
        print(f"  Wikipedia failed ({e}), using built-in S&P 500 list")
        # Comprehensive ~500 ticker fallback (current S&P 500 as of 2024)
        return [
            "MMM","AOS","ABT","ABBV","ACN","ADBE","AMD","AES","AFL","A","APD","AKAM","ALK","ALB",
            "ARE","ALGN","ALLE","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP",
            "AXP","AIG","AMT","AWK","AMP","AME","AMGN","APH","ADI","ANSS","AON","APA","AAPL",
            "AMAT","APTV","ACGL","ADM","ANET","AJG","AIZ","T","ATO","ADSK","ADP","AZO","AVB",
            "AVY","AXON","BKR","BALL","BAC","BAX","BDX","WRB","BBY","BIO","TECH","BIIB","BLK",
            "BX","BK","BA","BKNG","BSX","BMY","AVGO","BR","BRO","BF-B","BRK-B","BLDR","CHRW",
            "CDNS","CZR","CPT","CPB","COF","CAH","KMX","CCL","CARR","CTLT","CAT","CBOE","CBRE",
            "CDW","CE","COR","CNC","CNP","CF","CHTR","CVX","CMG","CB","CHD","CI","CINF","CTAS",
            "CSCO","C","CFG","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CAG","COP","ED","STZ",
            "CEG","COO","CPRT","GLW","CPAY","CTVA","CSGP","COST","CTRA","CCI","CSX","CMI","CVS",
            "DHR","DRI","DVA","DAY","DE","DAL","XRAY","DVN","DXCM","FANG","DLR","DFS","DG","DPZ",
            "DOV","DOW","DHI","DTE","DUK","DD","EMN","ETN","EBAY","ECL","EIX","EW","EA","ELV",
            "LLY","EMR","ENPH","ETR","EOG","EPAM","EQT","EFX","EQIX","EQR","ESS","EL","ETSY",
            "EG","EVRG","ES","EXC","EXPE","EXR","XOM","FFIV","FDS","FICO","FAST","FRT","FDX",
            "FIS","FITB","FSLR","FE","FI","FMC","F","FTNT","FTV","FOXA","FOX","BEN","FCX","GRMN",
            "IT","GE","GEHC","GEV","GEN","GNRC","GD","GIS","GM","GPC","GILD","GPN","GL","GDDY",
            "GS","HAL","HIG","HAS","HCA","DOC","HSIC","HSY","HES","HPE","HLT","HOLX","HD","HON",
            "HRL","HST","HWM","HPQ","HUBB","HUM","HBAN","HII","IBM","IEX","IDXX","ITW","INCY",
            "IR","PODD","INTC","ICE","IFF","IP","IPG","INTU","ISRG","IVZ","INVH","IQV","IRM",
            "JBHT","JBL","JKHY","J","JNJ","JCI","JPM","JNPR","K","KVUE","KDP","KEY","KEYS","KMB",
            "KIM","KMI","KLAC","KHC","KR","LHX","LH","LRCX","LW","LVS","LDOS","LEN","LNC","LIN",
            "LYV","LKQ","LMT","L","LOW","LULU","LYB","MTB","MRO","MPC","MKTX","MAR","MMC","MLM",
            "MAS","MA","MTCH","MKC","MCD","MCK","MDT","MRK","META","MET","MTD","MGM","MCHP","MU",
            "MSFT","MAA","MRNA","MHK","MOH","TAP","MDLZ","MPWR","MNST","MCO","MS","MOS","MSI",
            "MSCI","NDAQ","NTAP","NFLX","NEM","NWSA","NWS","NEE","NKE","NI","NDSN","NSC","NTRS",
            "NOC","NCLH","NRG","NUE","NVDA","NVR","NXPI","ORLY","OXY","ODFL","OMC","ON","OKE",
            "ORCL","OTIS","PCAR","PKG","PLTR","PANW","PARA","PH","PAYX","PAYC","PYPL","PNR","PEP",
            "PFE","PCG","PM","PSX","PNW","PNC","POOL","PPG","PPL","PFG","PG","PGR","PRU","PEG",
            "PTYD","PTC","PSA","PHM","QRVO","PWR","QCOM","DGX","RL","RJF","RTX","O","REG","REGN",
            "RF","RSG","RMD","RVTY","ROK","ROL","ROP","ROST","RCL","SPGI","CRM","SBAC","SLB",
            "STX","SRE","NOW","SHW","SPG","SWKS","SJM","SNA","SOLV","SO","LUV","SWK","SBUX","STT",
            "STLD","STE","SYK","SMCI","SYF","SNPS","SYY","TMUS","TROW","TTWO","TPR","TRGP","TGT",
            "TEL","TDY","TFX","TER","TSLA","TXN","TXT","TMO","TJX","TSCO","TT","TDG","TRV","TRMB",
            "TFC","TYL","TSN","USB","UBER","UDR","ULTA","UNP","UAL","UPS","URI","UNH","UHS","VLO",
            "VTR","VLTO","VRSN","VRSK","VZ","VRTX","VTRS","VICI","V","VST","VMC","WRK","WAB","WMT",
            "WBA","WM","WAT","WEC","WFC","WELL","WST","WDC","WY","WHR","WMB","WTW","GWW","WYNN",
            "XEL","XYL","YUM","ZBRA","ZBH","ZTS",
        ]


def download_prices(tickers, start=START, end=END, batch=200):
    """Batch download monthly adjusted close prices in chunks to avoid timeout."""
    all_prices = []
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i+batch]
        print(f"  Price batch {i//batch + 1}: {len(chunk)} tickers...")
        try:
            raw = yf.download(chunk, start=start, end=end, interval="1mo",
                              auto_adjust=True, progress=False, threads=True)
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"]
            else:
                prices = raw
            all_prices.append(prices)
        except Exception as e:
            print(f"    Batch failed: {e}")
    if not all_prices:
        return pd.DataFrame()
    prices = pd.concat(all_prices, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices.index = pd.to_datetime(prices.index)
    # Convert to month-end timestamps
    prices.index = prices.index + pd.offsets.MonthEnd(0)
    prices = prices[(prices.index >= start) & (prices.index <= end)]
    prices = prices.dropna(how="all")
    # Drop tickers with < 60 months of data (need rolling windows)
    prices = prices.loc[:, prices.notna().sum() >= 60]
    print(f"  Total prices: {prices.shape[0]} months × {prices.shape[1]} tickers")
    return prices


def main():
    print("=== STOCK-LEVEL PRICE DOWNLOAD ===\n")

    tickers = get_sp500_tickers()
    prices = download_prices(tickers)

    if prices.empty:
        print("ERROR: no price data downloaded")
        return

    prices.to_parquet(f"{DATA}/stock_prices_monthly.parquet")

    coverage = prices.notna().sum()
    print(f"\n=== PRICE DATA REPORT ===")
    print(f"Tickers:   {prices.shape[1]}")
    print(f"Months:    {prices.shape[0]}  ({prices.index.min():%Y-%m} to {prices.index.max():%Y-%m})")
    print(f"Avg months per ticker:  {coverage.mean():.0f}")
    print(f"Tickers with full 408m: {(coverage >= 408).sum()}")
    print(f"Tickers with >120m:     {(coverage >= 120).sum()}")
    print(f"\n*** SURVIVORSHIP BIAS: universe = current S&P 500 only ***")
    print(f"Data pipeline complete. {prices.shape[1]} tickers.")


if __name__ == "__main__":
    main()
