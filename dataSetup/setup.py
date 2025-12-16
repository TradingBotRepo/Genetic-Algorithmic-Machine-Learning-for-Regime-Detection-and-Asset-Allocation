import yfinance as yf
import pandas as pd
import os

def prepare_data():
    print("Downloading data...")
    
    tickers = [
        "XLK", "XLF", "XLV", "XLE", "XLY", "XLI", "XLU", "XLP", "XLB", 
        "GLD", "TLT"
    ]
    macro_tickers = ["^VIX", "^TNX"]
    all_tickers = tickers + macro_tickers

    print(f"Fetching {len(all_tickers)} tickers from 2000-01-01...")
    
    raw_data = yf.download(all_tickers, start="2000-01-01", auto_adjust=True, progress=True)

    try:
        df = raw_data['Close']
    except KeyError:
        df = raw_data

    df = df.dropna(axis=1, how='all')
    df = df.ffill().dropna()
    
    print(f"Data Range: {df.index[0]} to {df.index[-1]}")
    
    available_assets = [t for t in tickers if t in df.columns]
    available_macro = [t for t in macro_tickers if t in df.columns]
    
    final_df = df[available_assets + available_macro]
    
    final_df.reset_index(inplace=True)
    
    final_df.rename(columns={final_df.columns[0]: "Date"}, inplace=True)

    output_file = "market_data.csv"
    final_df.to_csv(output_file, index=False)
    print(f"Saved {len(final_df)} rows to {output_file}")
    print(f"Training Data (Pre-2015): {len(final_df[final_df['Date'] < '2015-01-01'])} days")

if __name__ == "__main__":
    prepare_data()