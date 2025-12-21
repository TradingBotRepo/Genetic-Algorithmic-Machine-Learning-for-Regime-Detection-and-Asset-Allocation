

#  Genetic Regime-Based Asset Allocator

This project is an **Artificial Intelligence (AI) trading system** written in Julia. It uses a **Genetic Algorithm (GA)** to evolve a "Brain" that manages a portfolio of 11 assets (Sector ETFs + Gold + Bonds).

Instead of a static strategy (like "buy and hold"), this system adapts to market conditions. It constantly monitors **Volatility (VIX)**, **Interest Rates (TNX)**, and **Price Trends** to switch between **8 different personality modes (Regimes)**.

---

##  1. Input Data (`market_data.csv`)

The system is hard-coded to read your specific file structure.

**File Structure:**

* **Column 1:** `Date` (The timeline)
* **Columns 2-12 (The Tradable Assets):**
* `XLK` (Technology), `XLF` (Financials), `XLV` (Health), `XLE` (Energy), `XLY` (Discretionary), `XLI` (Industrial), `XLU` (Utilities), `XLP` (Staples), `XLB` (Materials)
* `GLD` (Gold)
* `TLT` (Long-Term Bonds)


* **Column 13:** `^VIX` (Volatility Index - "The Fear Gauge")
* **Column 14:** `^TNX` (10-Year Treasury Yield - "Interest Rates")

---

##  2. How "The Brain" Works

The core of this system is a decision engine that runs every single trading day. It answers the question: **"What kind of world are we in today, and how should I invest?"**

### A. The Sensors (Inputs)

The brain reads 3 sensors from your data:

1. **Panic Sensor:** Is the `^VIX` (Col 13) too high?
2. **Rate Sensor:** Is the `^TNX` (Col 14) too high?
3. **Trend Sensor:** Is the price of `XLK` (Technology) above its moving average?
* *Note: The script uses the first asset column (`XLK`) as the "Trend Proxy" for the general market.*



### B. The Genome (DNA)

Each strategy is defined by a "Gene" (a list of numbers). A single Gene contains:

1. **VIX Threshold:** (e.g., `24.5`)
2. **MA Window:** (e.g., `185` days)
3. **TNX Threshold:** (e.g., `3.8%`)
4. **Portfolio Weights:** A massive list of percentages for every possible market scenario.

### C. The Processing Logic (8 Regimes)

Every day, the logic calculates a "Regime ID" (1 through 8) based on the sensors.

| Regime | Is Panic? | High Rates? | Tech Trend? | Description |
| --- | --- | --- | --- | --- |
| **1** | No | No | **Up** | **"Goldilocks Zone"** (Calm, cheap money, bull market) |
| **2** | No | No | **Down** | Correction in a calm market |
| **3** | No | **Yes** | **Up** | Inflationary Boom (Rates up, but stocks up) |
| **4** | No | **Yes** | **Down** | Rate-driven correction |
| **5** | **Yes** | No | **Up** | "Wall of Worry" (Fear is high, but price is rising) |
| **6** | **Yes** | No | **Down** | **Standard Crash** (Fear high, price dropping) |
| **7** | **Yes** | **Yes** | **Up** | Volatile Inflationary market |
| **8** | **Yes** | **Yes** | **Down** | **Total Crisis** (High rates + Panic + Crash) |

### D. The Action (Rebalancing)

Once the brain identifies the Regime (e.g., "Regime 8: Total Crisis"), it ignores all other instructions and loads the specific portfolio weights evolved for that exact scenario.

**Example of Adaptive Behavior:**

* **In Regime 1 (Goldilocks):** The brain might deploy `40% XLK`, `30% XLY`, `30% XLF` (Aggressive Growth).
* **In Regime 6 (Crash):** The brain might instantly switch to `0% Stocks`, `50% TLT`, `50% GLD` (Safety).

---

##  3. How It Learns (Genetic Algorithm)

The system does not know the "correct" weights beforehand. It learns them through trial and error, mimicking natural selection.

### The "Island" Model

The code runs 3 parallel simulations (Islands), each with a different goal:

1. **Aggressive Island:** Maximizes **Total Return** (CAGR).
2. **Balanced Island:** Maximizes **Sortino Ratio** (Return vs. Downside Risk).
3. **Conservative Island:** Penalizes Drawdowns heavily. It wants a smooth line up.

### The Training Loop

1. **Create:** It generates 300 random "Brains" (100 per island).
2. **Test:** It runs them through historical data (random 2-year chunks from 2004-2023).
3. **Kill:** The bottom 50% of performers are deleted.
4. **Breed:** The top performers "mate." Their genes (thresholds and weights) are mixed to create new children.
* *Mutation:* Occasionally, a number is randomly tweaked to discover new possibilities.


5. **Repeat:** This happens for hundreds of generations until a "Super Strategy" evolves.

---

##  4. Output Files

As the code runs, it saves its discoveries:

* **`best_strategy_metrics.csv`**: A spreadsheet of the best strategies found, showing their Annual Return, Max Drawdown, and Sharpe Ratio.
* **`best_genes_balanced.jsonl`**: The actual "Save File" of the best brains. You can open this text file to see exactly what weights the AI chose for each regime.
* **`best_equity_balanced_2015.csv`**: A CSV ready for Excel plotting. It shows exactly how $10,000 would have grown since 2015 using the best strategy.

---

##  5. Quick Start Guide

1. **Install Julia:** Download from [julialang.org](https://julialang.org).
2. **Install Packages:**
Open Julia and run:
```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "Statistics", "Random", "Serialization", "Dates", "JSON"])

```


3. **Run:**
Place your `market_data.csv` in the same folder as the script.
Run in terminal:
```bash
julia --threads auto your_script_name.jl

```


4. **Monitor:**
Watch the console. When you see a result you like (e.g., `BALANCED Score: 2.5`), you can stop the script (`Ctrl+C`) and check the generated `.csv` files.
