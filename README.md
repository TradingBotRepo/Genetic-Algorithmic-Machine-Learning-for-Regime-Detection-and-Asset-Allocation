
# Multi-Objective Regime-Switching Asset Allocator

## 1. Project Overview

This software implements a high-performance **Genetic Algorithm (GA)** for optimizing dynamic asset allocation strategies. Unlike static portfolio theories (e.g., Modern Portfolio Theory), this system utilizes a **Regime-Switching Framework**. It continuously monitors macroeconomic and technical indicators to classify the market environment into one of eight distinct states ("regimes") and adjusts portfolio weights accordingly.

The system utilizes an **Island Model Evolutionary Strategy**, evolving three distinct populations simultaneously to satisfy competing investment objectives: Aggressive (Maximum Return), Balanced (Risk-Adjusted Return), and Conservative (Drawdown Minimization).

---

## 2. System Architecture: "The Brain"

The core decision-making engine is a Finite State Machine (FSM) that dictates portfolio allocation based on environmental sensors.

### 2.1. Sensory Inputs

At each time step , the algorithm evaluates three binary conditions derived from the input data:

1. **Volatility Regime ():**
* *Input:* CBOE Volatility Index (`^VIX`).
* *Logic:* 
* *Significance:* Identifies periods of high market stress or fear.


2. **Interest Rate Environment ():**
* *Input:* 10-Year Treasury Yield (`^TNX`).
* *Logic:* 
* *Significance:* Identifies inflationary or tightening monetary environments which affect bond/equity correlations.


3. **Market Trend ():**
* *Input:* Primary Risk Asset Price (Technology Sector ETF).
* *Logic:* 
* *Significance:* Identifies the prevailing technical direction of the market using a variable Moving Average (MA).



### 2.2. The Genome Structure

Each candidate strategy is represented by a genome vector containing continuous floating-point values. The genome encodes both the threshold parameters and the allocation matrices:

* **Locus 1-3 (Thresholds):** , , .
* **Locus 4-N (Weight Matrix):** A flattened array defining specific portfolio weights for every possible regime.

### 2.3. Decision Logic (The 8 Regimes)

The three binary inputs create a state space of  unique regimes. The algorithm calculates the current state index at every time step:

| Regime ID | Volatility | Rate Environment | Market Trend | Market Characterization |
| --- | --- | --- | --- | --- |
| **1** | Low | Low | Uptrend | **Expansionary/Bull:** Favorable liquidity and growth. |
| **2** | Low | Low | Downtrend | **Correction:** Non-systemic market pullback. |
| **3** | Low | High | Uptrend | **Inflationary Growth:** Rising yields accompanied by growth. |
| **4** | Low | High | Downtrend | **Rate-Driven Weakness:** Valuation compression due to rates. |
| **5** | High | Low | Uptrend | **"Wall of Worry":** High volatility but resilient price action. |
| **6** | High | Low | Downtrend | **Deflationary Crash:** Typical crisis behavior. |
| **7** | High | High | Uptrend | **Volatile Inflation:** High uncertainty and yields. |
| **8** | High | High | Downtrend | **Stagflationary Crisis:** Systemic failure (High rates + Panic + Crash). |

Upon identifying the active regime, the system retrieves the associated weight vector, applies a **Softmax Normalization** to ensure unity (sum of weights = 1.0), and rebalances the portfolio.

---

## 3. Algorithmic Methodology

The optimization engine employs a parallelized genetic algorithm with the following characteristics:

### 3.1. The Island Model

To preserve diversity and optimize for the Efficient Frontier, the population is segregated into three "Islands" (sub-populations), each maximizing a different fitness function:

* **Aggressive Tribe:** Fitness = Annualized Return ().
* **Balanced Tribe:** Fitness = Sortino Ratio or Sharpe Ratio.
* **Conservative Tribe:** Fitness = Weighted Score prioritizing Max Drawdown () reduction.

### 3.2. Robustness via Stochastic Sampling

To prevent overfitting to a specific historical path, the fitness evaluation does not rely on a single simulation. Instead, it employs a **Monte Carlo approach**:

1. For every generation,  random start dates are selected.
2. Each agent is simulated over a fixed window (e.g., 504 trading days) starting from these dates.
3. **Robustness Metric:** The final score is the mean performance minus a penalty for variance:



This enforces the selection of strategies that perform consistently across varied temporal conditions.

---

## 4. Technical Justification: Julia vs. Python/PyTorch

This codebase is architected in **Julia** specifically to address computational bottlenecks inherent in regime-switching simulations. While Python frameworks like PyTorch are industry standards for Deep Learning, they are suboptimal for this specific algorithmic class.

### 4.1. The "Branching" Problem

Genetic Algorithms for trading involve discrete, path-dependent logic (e.g., `if state == 6 then rebalance`).

* **PyTorch Limitation:** Deep Learning frameworks optimize for dense matrix operations (SIMD) and require differentiable graphs for Gradient Descent. They struggle efficiently parallelizing complex branching logic ("Control Flow") across batches without significant overhead.
* **Julia Advantage:** Julia utilizes Just-In-Time (JIT) compilation via LLVM. It compiles high-level loop logic directly into optimized machine code. This allows for complex, non-vectorized `for-loops` containing conditional rebalancing logic to execute at speeds comparable to C++.

### 4.2. Derivative-Free Optimization

The objective function in algorithmic trading is non-differentiable and non-convex (a discrete "Buy" or "Sell" decision creates a discontinuity in the loss landscape).

* **PyTorch Limitation:** PyTorch is fundamentally an engine for Automatic Differentiation (Autograd). Since Genetic Algorithms do not use gradients, the overhead of the PyTorch graph construction provides no benefit.
* **Julia Advantage:** Julia provides a lightweight, high-throughput environment for raw numerical computation, allowing for millions of simulation steps per second without the latency of Python's interpreter or the overhead of a tensor graph.

---

## 5. Input Data Specification

The system requires a CSV file named `market_data.csv`. The `data_loader.py` script provided assists in fetching this data from Yahoo Finance.

**Required Schema:**

* **Row 1:** Header.
* **Column 1:** Date (`YYYY-MM-DD`).
* **Columns 2 through :** Tradable Assets (e.g., XLK, XLF, TLT, GLD).
* **Column :** Volatility Index (`^VIX`).
* **Column :** Interest Rate Index (`^TNX`).

**Current Asset Universe:**
The model is currently configured for US Sector ETFs (`XLK`, `XLF`, `XLV`, etc.), Gold (`GLD`), and Long-Term Treasuries (`TLT`).

---

## 6. Installation and Execution

### Prerequisites

* **Julia 1.6+**
* **Python 3.8+** (Only required for initial data fetching)

### Setup

1. **Initialize Data:**
Run the Python data loader to generate the required CSV.
```bash
python data_loader.py

```


2. **Install Julia Dependencies:**
Launch the Julia REPL and execute:
```julia
using Pkg
Pkg.add(["CSV", "DataFrames", "Statistics", "Random", "Serialization", "Dates", "JSON"])

```


3. **Execute Optimization:**
Run the optimizer using multi-threading for Island parallelization.
```bash
julia --threads auto trainog2.jl

```



---

## 7. Output Artifacts

The system generates the following outputs during execution:

* **`checkpoint_islands.jls`**: A serialized binary file containing the state of the entire population. Allows for pausing and resuming training.
* **`best_strategy_metrics.csv`**: A tabular log recording the performance metrics (Sharpe, Sortino, CAGR, Volatility) of every new optimal strategy discovered.
* **`best_genes_[type].jsonl`**: JSON Lines files containing the human-readable parameters and weight matrices for the best strategies in each tribe.
* **`best_equity_[type]_[context].csv`**: Time-series CSV files containing the daily account balance curves for the best strategies, suitable for external visualization or auditing.
