import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import os

#
def performance_metrics(equity: pd.Series):
    """Calculate key performance metrics for an equity curve"""
    equity_np = np.asarray(equity, dtype=float).reshape(-1)
    if equity_np.size < 2:
        return {
            "annual_return": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_dd": 0.0,
            "annual_vol": 0.0,
            "end_val": float(equity_np[-1]) if equity_np.size else 0.0
        }

    returns = np.diff(equity_np) / equity_np[:-1]
    start_val = equity_np[0]
    end_val = equity_np[-1]

    # CAGR
    cagr = 0.0
    if start_val > 0:
        cagr = (end_val / start_val) ** (252 / len(equity_np)) - 1.0
    # Annualized volatility
    ret_std = np.nan_to_num(returns.std(ddof=0), nan=0.0)
    ann_vol = ret_std * np.sqrt(252)

    # Sharpe
    sharpe = 0.0
    if ret_std > 1e-9:
        sharpe = (returns.mean() * 252) / (ret_std * np.sqrt(252))

    # Sortino
    downside = returns[returns < 0]
    sortino = 0.0
    down_std = np.nan_to_num(downside.std(ddof=0), nan=0.0)
    if downside.size > 0 and down_std > 1e-9:
        sortino = (returns.mean() * 252) / (down_std * np.sqrt(252))

    # Max Drawdown
    roll_max = np.maximum.accumulate(equity_np)
    valid_mask = roll_max > 0
    drawdown = np.zeros_like(equity_np)
    drawdown[valid_mask] = (equity_np[valid_mask] / roll_max[valid_mask]) - 1.0
    max_dd = drawdown.min()

    return {
        "annual_return": cagr,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_dd": max_dd,
        "annual_vol": ann_vol,
        "end_val": end_val
    }

def load_strategy(strategy_name):
    """Load equity curve for a specific strategy"""
    # Map 'hyper_aggressive' to the uploaded file
    if strategy_name == "hyper_aggressive":
        csv_file = "best_equity_2015.csv"
    else:
        csv_file = f"best_equity_{strategy_name}_2015.csv"
    
    if not os.path.exists(csv_file):
        return None
    
    try:
        df = pd.read_csv(csv_file, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
        
        # Robust column finding
        col = None
        if "Strategy" in df.columns:
            col = df["Strategy"]
        elif "equity" in df.columns:
            col = df["equity"]
        else:
            col = df.iloc[:, 0]
            
        return pd.to_numeric(col, errors='coerce').dropna()
            
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")
        return None


def plot_all_strategies():
    print("\n" + "="*70)
    print("GENERATING CHART WITH COMPACT INFO BOX")
    print("="*70)
    
    strategies = {}
    strategy_keys = ["hyper_aggressive", "aggressive", "balanced", "conservative", "non_ai"]
    
    colors = {
        "hyper_aggressive": "#FF0000", # Red
        "aggressive": "#FF8800",       # Orange 
        "balanced": "#0052cc",         # Blue
        "conservative": "#00AA00",     # Green
        "non_ai": "#9932CC",           # Purple
        "spy": "#888888"               # Gray
    }
    

    for key in strategy_keys:
        equity = load_strategy(key)
        if equity is not None:
            strategies[key] = equity
            print(f"Loaded {key.replace('_', ' ').upper()}")
    
    if not strategies:
        print("\nERROR: No strategies found!")
        return
    
    # --- Setup SPY Benchmark ---
    first_key = list(strategies.keys())[0]
    start_date = strategies[first_key].index[0]
    end_date = strategies[first_key].index[-1]
    
    spy_equity = None
    try:
        print("Downloading SPY benchmark...")
        spy_df = yf.download("SPY", start=start_date, end=end_date + pd.Timedelta(days=5), progress=False)
        
        if isinstance(spy_df.columns, pd.MultiIndex):
            try: spy = spy_df["Close"]
            except: spy = spy_df.iloc[:, 0]
        elif "Close" in spy_df.columns:
            spy = spy_df["Close"]
        else:
            spy = spy_df
            
        if isinstance(spy, pd.DataFrame): spy = spy.squeeze()
        
        spy = spy.reindex(strategies[first_key].index).ffill()
        
        # Normalize SPY
        init_cash = strategies[first_key].iloc[0]
        if len(spy) > 0 and spy.iloc[0] > 0:
            spy_equity = (spy / spy.iloc[0]) * init_cash
        else:
            spy_equity = pd.Series(init_cash, index=strategies[first_key].index)
            
    except Exception as e:
        print(f"Warning: SPY download failed ({e})")

    # --- Calculate Metrics & Global Ranges ---
    all_metrics = {}
    all_values = []
    
    # Process Strategies
    for key in strategies:
        all_metrics[key] = performance_metrics(strategies[key])
        all_values.extend(strategies[key].values)
        
    # Process SPY
    if spy_equity is not None:
        all_metrics["spy"] = performance_metrics(spy_equity)
        all_values.extend(spy_equity.values)

    # Calculate Fixed Y-Axis Range
    y_min, y_max = min(all_values), max(all_values)
    y_range = [y_min * 0.9, y_max * 1.1] # 10% padding
    if y_range[0] <= 0: y_range[0] = 1000 # Safety for log scale if needed

    # --- Build Plot ---
    fig = go.Figure()
    added_traces = []
    
    # Add Strategy Traces
    for key in strategy_keys:
        if key in strategies:
            d_name = key.upper().replace("_", " ")
            fig.add_trace(go.Scatter(
                x=strategies[key].index,
                y=strategies[key],
                mode='lines',
                name=d_name,
                line=dict(color=colors.get(key, "black"), width=2.5),
                hovertemplate=f'{d_name}<br>Date: %{{x|%Y-%m-%d}}<br>Val: $%{{y:,.0f}}<extra></extra>'
            ))
            added_traces.append(d_name)

    # Add SPY Trace
    if spy_equity is not None:
        fig.add_trace(go.Scatter(
            x=spy_equity.index,
            y=spy_equity,
            mode='lines',
            name='SPY Benchmark',
            line=dict(color=colors["spy"], width=1.5, dash='dot'),
            hovertemplate='SPY<br>Date: %{x|%Y-%m-%d}<br>Val: $%{y:,.0f}<extra></extra>'
        ))
        added_traces.append("SPY Benchmark")

    # --- UI: Stats Sheet (top-left) ---
    header = "NAME           Ret%   Vol%   Shrp Sort  MaxDD    End$"
    stats_lines = [header]

    def stats_row(label: str, metrics: dict) -> str:
        """Format a single line for the stats box; spaces converted later to nbsp."""
        label = (label[:14]).ljust(14)
        return (
            f"{label}"
            f"{metrics['annual_return']*100:6.1f}% "
            f"{metrics['annual_vol']*100:6.1f}% "
            f"{metrics['sharpe']:5.2f} "
            f"{metrics['sortino']:5.2f} "
            f"{metrics['max_dd']*100:7.1f}% "
            f"${metrics['end_val']:,.0f}"
        )

    # Add Strategies to stats box
    for key in strategy_keys:
        if key in strategies:
            d_name = key.upper().replace("_", " ")
            if "HYPER" in d_name:
                d_name = "HYPER-AGGR"
            if "NON" in d_name:
                d_name = "NON-AI BOT"
            stats_lines.append(stats_row(d_name, all_metrics[key]))

    # Add SPY to stats box
    if spy_equity is not None and "spy" in all_metrics:
        stats_lines.append(stats_row("SPY", all_metrics["spy"]))

    # Convert spaces to &nbsp; so alignment is preserved in the annotation
    stats_text_lines = [line.replace(" ", "&nbsp;") for line in stats_lines]
    stats_text = "<b>STATS&nbsp;(Annualized)</b><br>" + "<br>".join(stats_text_lines)

    # Add Annotation (Top Left, Anchored)
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.99,
        text=stats_text,
        showarrow=False,
        align="left",
        bgcolor="rgba(255, 255, 255, 0.98)",
        bordercolor="#333",
        borderwidth=1,
        borderpad=8,
        font=dict(size=12, family="Arial")
    )

    # --- UI: Dropdown Menus ---
    buttons = []
    # "All" Button
    buttons.append(dict(label="Show All", method="update", 
                        args=[{"visible": [True] * len(added_traces)},
                              {"title": "All Strategies"}]))
    
    # Individual Buttons
    for i, name in enumerate(added_traces):
        vis = [False] * len(added_traces)
        vis[i] = True
        buttons.append(dict(label=f"Only {name}", method="update", 
                            args=[{"visible": vis},
                                  {"title": f"Strategy: {name}"}]))

    fig.update_layout(
        updatemenus=[dict(type="dropdown", direction="down", x=1.0, y=1.05, 
                          showactive=True, active=0, buttons=buttons)]
    )

    # --- Final Layout (Fixed Axes) ---
    fig.update_layout(
        title="Strategy Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        template="plotly_white",
        height=700,
        margin=dict(t=50, l=50, r=50, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        hovermode="x unified",
        
        # FIXED AXES LOGIC
        yaxis=dict(
            range=y_range,
            fixedrange=False
        )
    )
    
    print("\n" + "="*100)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("="*100)
    print(f"{'Strategy':<20} {'Annual Return':>15} {'Annual Vol':>15} {'Sharpe':>10} {'Sortino':>10} {'Max DD':>10} {'End Value':>15}")
    print("-"*100)
    for key in strategy_keys:
        if key in strategies:
            d_name = key.upper().replace("_", " ")
            if "HYPER" in d_name:
                d_name = "HYPER-AGGR"
            if "NON" in d_name:
                d_name = "NON-AI BOT"
            m = all_metrics[key]
            print(f"{d_name:<20} {m['annual_return']*100:>14.2f}% {m['annual_vol']*100:>14.2f}% {m['sharpe']:>10.2f} {m['sortino']:>10.2f} {m['max_dd']*100:>9.2f}% ${m['end_val']:>14,.0f}")
    if spy_equity is not None and "spy" in all_metrics:
        m = all_metrics["spy"]
        print(f"{'SPY':<20} {m['annual_return']*100:>14.2f}% {m['annual_vol']*100:>14.2f}% {m['sharpe']:>10.2f} {m['sortino']:>10.2f} {m['max_dd']*100:>9.2f}% ${m['end_val']:>14,.0f}")
    print("="*100 + "\n")
    print("Opening interactive plot...")
    fig.show()

if __name__ == "__main__":
    plot_all_strategies()
