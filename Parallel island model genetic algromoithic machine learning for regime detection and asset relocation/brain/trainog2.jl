using CSV
using DataFrames
using Statistics
using Random
using Serialization
using Dates
using Base.Threads
using JSON

#Keyword:GA system controls 
const resetSimulation = false
const popPerObjective = 100
const eliteCount       = 7
const graphStartDate = Date(2015, 1, 1)
const testStartDate  = Date(2024, 1, 1)
const samplesPerGen  = 18
const windowSize      = 504

#Out put files
const checkpointFile = "checkpoint_islands.jls"
const equityFileAggressive2015     = "best_equity_aggressive_2015.csv"
const equityFileBalanced2015       = "best_equity_balanced_2015.csv"
const equityFileConservative2015   = "best_equity_conservative_2015.csv"
const equityFileAggressiveTrain    = "best_equity_aggressive_train.csv"
const equityFileBalancedTrain      = "best_equity_balanced_train.csv"
const equityFileConservativeTrain  = "best_equity_conservative_train.csv"
const equityFileAggressiveTest     = "best_equity_aggressive_test.csv"
const equityFileBalancedTest       = "best_equity_balanced_test.csv"
const equityFileConservativeTest   = "best_equity_conservative_test.csv"
const jsonlAggressive   = "best_genes_aggressive.jsonl"
const jsonlBalanced     = "best_genes_balanced.jsonl"
const jsonlConservative = "best_genes_conservative.jsonl"
const metricsFile     = "best_strategy_metrics.csv"



# sortino or sharpe 
const balancedUses = :sortino 

# Conservative weighting
const consWDd      = 80.0
const consWSortino = 1.0
const consWSharpe  = 1.0
const consWCalmar  = 1.0


const robustK = 1.0

const ruinDdLimit = 0.99

const useSoftmaxWeights = false


if resetSimulation
    if isfile(checkpointFile)
        println(" Resetting Simulation Deleting checkpoints")
        rm(checkpointFile, force=true)
    end


    for file in (
        equityFileAggressive2015, equityFileBalanced2015, equityFileConservative2015,
        equityFileAggressiveTrain, equityFileBalancedTrain, equityFileConservativeTrain,
        equityFileAggressiveTest, equityFileBalancedTest, equityFileConservativeTest,
        metricsFile, jsonlAggressive, jsonlBalanced, jsonlConservative
    )
        if isfile(file)
            rm(file, force=true)
            println("Deleted existing file: $file")
        end
    end
else
    println(" Continuing from last checkpoint")
end

println("Loading data")
df = CSV.read("market_data.csv", DataFrame)

const dateCol   = df[:, 1]
const totalRows = size(df, 1)

graph_idx = findfirst(d -> d >= graphStartDate, dateCol)
if isnothing(graph_idx) || graph_idx < 100
    println("Erroe: Data insufficient.")
    exit()
end

train_end_idx = findfirst(d -> d >= testStartDate, dateCol)
if isnothing(train_end_idx)
    println("Error: TEST_START_DATE not found in data. Using full dataset.")
    train_end_idx = totalRows + 1
end
train_end_idx -= 1

const macroCount = 2
const totalCols = size(df, 2) - 1
const nAssets = totalCols - macroCount
const nRegimes = 8


const geneLength = 3 + (nAssets * nRegimes)

dataMatrix = Matrix(df[:, 2:end])

const trainPrices = dataMatrix[1:train_end_idx, 1:nAssets]
const trainMacro = dataMatrix[1:train_end_idx, (nAssets+1):end]
const trainDates = dateCol[1:train_end_idx]

const testPrices = dataMatrix[train_end_idx+1:end, 1:nAssets]
const testMacro = dataMatrix[train_end_idx+1:end, (nAssets+1):end]
const testDates = dateCol[train_end_idx+1:end]

const allPrices = dataMatrix[:, 1:nAssets]
const allMacro  = dataMatrix[:, (nAssets+1):end]
const allDates = dateCol

const assetNames = names(df)[2:nAssets+1]


println("GA island model")

println("Training Period: $(trainDates[1]) to $(trainDates[end])")
println("Training Days:   $(length(trainDates))")
if !isempty(testDates)
    println("Testing Period:  $(testDates[1]) to $(testDates[end])")
    println("Testing Days:    $(length(testDates))")
end
println("\nObjectives (Running in Parallel Islands):")


println()

@inline function safe_std(v::AbstractVector{Float64})
    s = std(v)
    return isnan(s) ? 0.0 : s
end

function softmax!(x::AbstractVector{Float64})
    m = maximum(x)
    s = 0.0
    @inbounds for i in eachindex(x)
        xi = exp(x[i] - m)
        x[i] = xi
        s += xi
    end
    invs = s > 0 ? 1.0 / s : 0.0
    @inbounds for i in eachindex(x)
        x[i] *= invs
    end
    return x
end


function gene_to_regime_weights!(rw::Matrix{Float64}, gene::Vector{Float64}, n_assets::Int)

    w_start = 4
    all_weights = @view gene[w_start:end]

    @inbounds for r in 0:(nRegimes-1)
        idx_s = 1 + (r * n_assets)
        idx_e = idx_s + n_assets - 1
        raw_w = @view all_weights[idx_s:idx_e]

        if useSoftmaxWeights
            rw_row = @view rw[r+1, :]
            rw_row .= raw_w
            softmax!(rw_row)
        else
            s = sum(raw_w)
            if s == 0.0
                rw[r+1, :] .= 1.0 / n_assets
            else
                invs = 1.0 / s
                rw[r+1, :] .= raw_w .* invs
            end
        end
    end
    return rw
end

@inline function objective_score(obj::Symbol,
                                 ann_ret::Float64,
                                 sortino::Float64,
                                 sharpe::Float64,
                                 calmar::Float64,
                                 max_dd::Float64)
    dd_abs = abs(max_dd)

    if dd_abs > ruinDdLimit
        return -1e9
    end

    if obj === :aggressive
        return ann_ret
    elseif obj === :balanced
        return balancedUses === :sortino ? sortino : sharpe
    elseif obj === :conservative
         return (-consWDd * dd_abs) +
             (consWSortino * sortino) +
             (consWSharpe  * sharpe) +
             (consWCalmar  * calmar)
    else
        return -1e9
    end
end

function simulate_strategy_fast(price_slice::AbstractMatrix{Float64},
                                macro_slice::AbstractMatrix{Float64},
                                gene::Vector{Float64};
                                return_equity::Bool=false,
                                return_regime_counts::Bool=false)

    n_days, n_assets = size(price_slice)

    vix_thresh = gene[1]
    ma_window  = round(Int, gene[2])
    tnx_thresh = gene[3]

    if n_days < ma_window + 5
        return (0.0, -1.0, 0.0, 0.0, 0.0, 0.0, nothing, nothing)
    end

    tech_prices = @view price_slice[:, 1]


    rw = Matrix{Float64}(undef, nRegimes, n_assets)
    gene_to_regime_weights!(rw, gene, n_assets)


    equity_curve = return_equity ? Vector{Float64}(undef, n_days - (ma_window + 1)) : nothing
    regime_counts = return_regime_counts ? zeros(Int, nRegimes) : nothing


    cash     = 10_000.0
    holdings = zeros(Float64, n_assets)


    sum_ma = 0.0
    @inbounds for i in 1:ma_window
        sum_ma += tech_prices[i]
    end


    n_ret = 0
    mean_r = 0.0
    m2_r   = 0.0


    n_down = 0
    mean_d = 0.0
    m2_d   = 0.0


    peak_equity = cash
    max_dd = 0.0


    prev_value = cash

    out_idx = 0

    @inbounds for t in (ma_window + 1):n_days

        day_value = cash + sum(holdings .* @view(price_slice[t, :]))


        if return_equity
            out_idx += 1
            equity_curve[out_idx] = day_value
        end


        if t > (ma_window + 1)
            r = (day_value - prev_value) / prev_value

            n_ret += 1
            δ = r - mean_r
            mean_r += δ / n_ret
            m2_r   += δ * (r - mean_r)

            if r < 0.0
                n_down += 1
                δd = r - mean_d
                mean_d += δd / n_down
                m2_d   += δd * (r - mean_d)
            end


            if day_value > peak_equity
                peak_equity = day_value
            else
                dd = (day_value - peak_equity) / peak_equity
                if dd < max_dd
                    max_dd = dd
                end
            end


            if abs(max_dd) > ruinDdLimit
                return (0.0, max_dd, 0.0, 0.0, 0.0, 0.0, return_equity ? equity_curve[1:out_idx] : nothing, regime_counts)
            end
        end

        prev_idx = t - 1


        ma_val = sum_ma / ma_window

        prev_vix   = macro_slice[prev_idx, 1]
        prev_tnx   = macro_slice[prev_idx, 2]
        prev_price = tech_prices[prev_idx]

        is_panic     = prev_vix > vix_thresh
        is_downtrend = prev_price < ma_val
        is_highrates = prev_tnx > tnx_thresh

        regime_idx = (Int(is_panic) * 4) + (Int(is_highrates) * 2) + Int(is_downtrend) + 1

        if return_regime_counts
            regime_counts[regime_idx] += 1
        end


        target_w = @view rw[regime_idx, :]
        holdings .= (day_value .* target_w) ./ @view(price_slice[t, :])
        cash = 0.0

        prev_value = day_value


        if t < n_days
            sum_ma += tech_prices[t] - tech_prices[t - ma_window]
        end
    end

    if n_ret < 5
        return (0.0, max_dd, 0.0, 0.0, 0.0, 0.0, equity_curve, regime_counts)
    end

    var_r = n_ret > 1 ? (m2_r / (n_ret - 1)) : 0.0
    vol = sqrt(max(var_r, 0.0)) * sqrt(252)

    ann_ret = mean_r * 252
    sharpe = vol > 1e-12 ? (ann_ret / vol) : 0.0

    var_down = n_down > 1 ? (m2_d / (n_down - 1)) : 0.0
    std_down = sqrt(max(var_down, 0.0))
    sortino = std_down > 1e-12 ? (ann_ret / (std_down * sqrt(252))) : 0.0

    dd_abs = abs(max_dd)
    calmar = dd_abs > 1e-12 ? (ann_ret / dd_abs) : 0.0

    return (ann_ret, max_dd, sortino, sharpe, calmar, vol,
            return_equity ? equity_curve : nothing,
            regime_counts)
end





function create_gene()
    g = rand(Float64, geneLength)
    g[1] = (g[1] * 30) + 10
    g[2] = (g[2] * 230) + 20
    g[3] = (g[3] * 6) + 1

    if !useSoftmaxWeights
        g[4:end] .= max.(g[4:end], 0.0)
    end

    return g
end


function mutate(gene::Vector{Float64}, rate::Float64)
    child = copy(gene)
    mask = rand(Float64, length(child)) .< rate
    noise = randn(length(child)) .* 0.15

    child[mask] .+= noise[mask]

    child[1] = clamp(child[1], 5.0, 60.0)
    child[2] = clamp(child[2], 10.0, 300.0)
    child[3] = clamp(child[3], 0.5, 10.0)
    child[4:end] = max.(child[4:end], 0.0)
    return child
end

function crossover(p1::Vector{Float64}, p2::Vector{Float64})
    child = copy(p1)
    mask = rand(Bool, length(child))
    child[mask] = p2[mask]
    return child
end





function append_metrics_row(strategy_name::String, gene::Vector{Float64},
                            train_metrics::NamedTuple, test_metrics::NamedTuple)
    row = DataFrame(
        Timestamp    = [string(now())],
        Strategy     = [strategy_name],
        Train_AnnRet = [train_metrics.ann_ret],
        Train_MaxDD  = [train_metrics.max_dd],
        Train_Sortino= [train_metrics.sortino],
        Train_Sharpe = [train_metrics.sharpe],
        Train_Calmar = [train_metrics.calmar],
        Train_Vol    = [train_metrics.vol],
        Test_AnnRet  = [test_metrics.ann_ret],
        Test_MaxDD   = [test_metrics.max_dd],
        Test_Sortino = [test_metrics.sortino],
        Test_Sharpe  = [test_metrics.sharpe],
        Test_Calmar  = [test_metrics.calmar],
        Test_Vol     = [test_metrics.vol],
        VIX_Thresh   = [gene[1]],
        MA_Window    = [round(Int, gene[2])],
        TNX_Thresh   = [gene[3]]
    )

    if isfile(metricsFile)
        existing = CSV.read(metricsFile, DataFrame)
        CSV.write(metricsFile, vcat(existing, row))
    else
        CSV.write(metricsFile, row)
    end
end

function save_best_gene_jsonl(strategy_name::String, gene::Vector{Float64}, train_metrics::NamedTuple, test_metrics::NamedTuple, robust_score::Float64)
    filename = strategy_name == "aggressive" ? jsonlAggressive :
               strategy_name == "balanced"   ? jsonlBalanced   :
                                              jsonlConservative


    w_start = 4
    n_assets = length(assetNames)
    regime_weights_dict = Dict()
    regime_names = [
        "Calm_LowRates_Uptrend",   "Calm_LowRates_Downtrend",
        "Calm_HighRates_Uptrend",  "Calm_HighRates_Downtrend",
        "Panic_LowRates_Uptrend",  "Panic_LowRates_Downtrend",
        "Panic_HighRates_Uptrend", "Panic_HighRates_Downtrend"
    ]

    all_weights = gene[w_start:end]
    for r in 0:(nRegimes-1)
        idx_s = 1 + (r * n_assets)
        idx_e = idx_s + n_assets - 1
        raw_w = all_weights[idx_s:idx_e]


        final_w = zeros(Float64, n_assets)
        if useSoftmaxWeights
            final_w .= raw_w
            softmax!(final_w)
        else
            s = sum(raw_w)
            if s == 0
                final_w .= 1.0 / n_assets
            else
                final_w .= raw_w ./ s
            end
        end

        regime_weights_dict[regime_names[r+1]] = Dict(assetNames[k] => final_w[k] for k in 1:n_assets)
    end

    data = Dict(
        "timestamp" => string(now()),
        "strategy" => strategy_name,
        "robust_score" => robust_score,
        "parameters" => Dict(
            "vix_threshold" => gene[1],
            "ma_window" => round(Int, gene[2]),
            "tnx_threshold" => gene[3]
        ),
        "train_metrics" => Dict(pairs(train_metrics)),
        "test_metrics" => Dict(pairs(test_metrics)),
        "weights" => regime_weights_dict,
        "raw_gene" => gene
    )

    open(filename, "a") do f
        JSON.print(f, data)
        println(f, "")
    end
end

function save_strategy(gene::Vector{Float64}, strategy_name::String, train_dates, test_dates, robust_score::Float64)

    all_ann, all_dd, all_sort, all_sharpe, all_calmar, all_vol, all_curve, _ =
        simulate_strategy_fast(allPrices, allMacro, gene; return_equity=true, return_regime_counts=false)


    tr_ann, tr_dd, tr_sort, tr_sharpe, tr_calmar, tr_vol, _, _ =
        simulate_strategy_fast(trainPrices, trainMacro, gene; return_equity=false, return_regime_counts=false)


    te_ann, te_dd, te_sort, te_sharpe, te_calmar, te_vol, _, _ =
        if !isempty(testPrices)
            simulate_strategy_fast(testPrices, testMacro, gene; return_equity=false, return_regime_counts=false)
        else
            (0.0, -1.0, 0.0, 0.0, 0.0, 0.0, nothing, nothing)
        end

    train_metrics = (ann_ret=tr_ann, max_dd=tr_dd, sortino=tr_sort, sharpe=tr_sharpe, calmar=tr_calmar, vol=tr_vol)
    test_metrics  = (ann_ret=te_ann, max_dd=te_dd, sortino=te_sort, sharpe=te_sharpe, calmar=te_calmar, vol=te_vol)

    append_metrics_row(strategy_name, gene, train_metrics, test_metrics)


    save_best_gene_jsonl(strategy_name, gene, train_metrics, test_metrics, robust_score)

    ma_window = round(Int, gene[2])
    start_offset = ma_window + 1

    file_2015 = strategy_name == "aggressive" ? equityFileAggressive2015 :
                strategy_name == "balanced"   ? equityFileBalanced2015   :
                                               equityFileConservative2015

    file_train = strategy_name == "aggressive" ? equityFileAggressiveTrain :
                 strategy_name == "balanced"   ? equityFileBalancedTrain   :
                                                equityFileConservativeTrain

    file_test  = strategy_name == "aggressive" ? equityFileAggressiveTest :
                 strategy_name == "balanced"   ? equityFileBalancedTest   :
                                                equityFileConservativeTest


    if length(allDates) >= start_offset && all_curve !== nothing && length(all_curve) > 0
        valid_dates = allDates[start_offset:end]
        n = min(length(valid_dates), length(all_curve))
        full_df = DataFrame(Date=valid_dates[1:n], Strategy=all_curve[1:n])

        mask_2015 = full_df.Date .>= graphStartDate
        df_2015 = full_df[mask_2015, :]
        if !isempty(df_2015)
            start_val = df_2015.Strategy[1]
            df_2015.Strategy = (df_2015.Strategy ./ start_val) .* 10000.0
            CSV.write(file_2015, df_2015)
        end
    end


    if length(train_dates) >= start_offset
        train_dates_slice = train_dates[start_offset:end]
        _, _, _, _, _, _, train_curve, _ =
            simulate_strategy_fast(trainPrices, trainMacro, gene; return_equity=true, return_regime_counts=false)
        if train_curve !== nothing
            n = min(length(train_dates_slice), length(train_curve))
            CSV.write(file_train, DataFrame(Date=train_dates_slice[1:n], Strategy=train_curve[1:n]))
        end
    end


    if !isempty(testPrices) && length(test_dates) >= start_offset
        test_dates_slice = test_dates[start_offset:end]
        _, _, _, _, _, _, test_curve, _ =
            simulate_strategy_fast(testPrices, testMacro, gene; return_equity=true, return_regime_counts=false)
        if test_curve !== nothing
            n = min(length(test_dates_slice), length(test_curve))
            CSV.write(file_test, DataFrame(Date=test_dates_slice[1:n], Strategy=test_curve[1:n]))
        end
    end

    return train_metrics, test_metrics
end





function main()
    objectives = (:aggressive, :balanced, :conservative)
    obj_names  = Dict(:aggressive=>"aggressive", :balanced=>"balanced", :conservative=>"conservative")

    populations = Dict{Symbol, Vector{Vector{Float64}}}()
    for obj in objectives
        populations[obj] = [create_gene() for _ in 1:popPerObjective]
    end

    gen = 0
    stagnation_counters = Dict{Symbol, Int}(obj => 0 for obj in objectives)

    best_by_obj = Dict{Symbol, Tuple{Float64, Union{Nothing, Vector{Float64}}}}(
        obj => (-Inf, nothing) for obj in objectives
    )

    if isfile(checkpointFile)
        println("Loading checkpoint...")
        try
            data = deserialize(checkpointFile)
            loaded_pops = get(data, :populations, nothing)
            gen = get(data, :gen, 0)
            loaded_best = get(data, :best_by_obj, nothing)

            if loaded_pops !== nothing
                populations = loaded_pops
            end
            if loaded_best !== nothing
                for obj in objectives
                    if haskey(loaded_best, obj)
                        best_by_obj[obj] = loaded_best[obj]
                    end
                end
            end
            println("Resuming at Gen $gen")
        catch e
            println("Warning: Checkpoint load failed ($e). Starting fresh.")
        end
    end

    max_train_start = size(trainPrices, 1) - windowSize - 1
    if max_train_start < 1
        println("ERROR: Insufficient training data.")
        return
    end

    println("\nStarting ISLAND-BASED genetic optimization (FAST sim)...")
    println("Tribes: $(length(objectives)) | Pop per Tribe: $popPerObjective | Total Agents: $(length(objectives)*popPerObjective)\n")

    while true
        gen += 1
        status_updates = String[]

        for obj in objectives
            pop = populations[obj]
            stag = stagnation_counters[obj]

            base_rate = stag > 30 ? 0.70 : (stag > 15 ? 0.40 : 0.18)
            oscillation = 0.1 * sin(gen * 0.1)
            current_rate = clamp(base_rate + oscillation, 0.1, 0.9)

            starts = rand(1:max_train_start, samplesPerGen)
            robust_scores = zeros(Float64, popPerObjective)

            @threads for idx in 1:popPerObjective
                sb = zeros(Float64, samplesPerGen)
                gene = pop[idx]

                @inbounds for k in 1:samplesPerGen
                    st = starts[k]
                    p_view = @view trainPrices[st:st+windowSize, :]
                    m_view = @view trainMacro[st:st+windowSize, :]

                    ann, dd, sort, sharpe, calmar, vol, _, _ =
                        simulate_strategy_fast(p_view, m_view, gene; return_equity=false, return_regime_counts=false)

                    sb[k] = objective_score(obj, ann, sort, sharpe, calmar, dd)
                end

                mu = mean(sb)
                sigma = safe_std(sb)
                robust_scores[idx] = mu - (robustK * sigma)
            end

            best_idx = argmax(robust_scores)
            best_val = robust_scores[best_idx]

            if best_val > best_by_obj[obj][1]
                best_by_obj[obj] = (best_val, pop[best_idx])


                tr_m, te_m = save_strategy(pop[best_idx], obj_names[obj], trainDates, testDates, best_val)


                print("\r" * " "^100 * "\r")
                println(" new [$(uppercase(obj_names[obj]))] | Score: $(round(best_val, digits=3)) | Train: $(round(tr_m.ann_ret*100,digits=1))% / $(round(tr_m.max_dd*100,digits=1))% DD")

                stagnation_counters[obj] = 0
            else
                stagnation_counters[obj] += 1
            end

            push!(status_updates, "$(uppercase(string(obj)[1:3]))=$(round(best_val, digits=2))")


            perm = sortperm(robust_scores, rev=true)
            elites = [pop[i] for i in perm[1:eliteCount]]

            if best_by_obj[obj][2] !== nothing
                push!(elites, best_by_obj[obj][2])
            end

            gene_hash(g) = begin
                h = UInt(0)
                @inbounds for x in g
                    h = xor(h, reinterpret(UInt, x))
                    h *= 0x9e3779b97f4a7c15
                end
                h
            end

            unique_elites = unique(g -> gene_hash(g), elites)
            if length(unique_elites) > eliteCount
                unique_elites = unique_elites[1:eliteCount]
            end

            new_pop = Vector{Vector{Float64}}(undef, 0)
            append!(new_pop, unique_elites)

            while length(new_pop) < popPerObjective
                if rand() < 0.70 && length(unique_elites) >= 2
                    p1 = unique_elites[rand(1:length(unique_elites))]
                    p2 = unique_elites[rand(1:length(unique_elites))]
                    child = crossover(p1, p2)
                else
                    child = copy(unique_elites[rand(1:length(unique_elites))])
                end
                push!(new_pop, mutate(child, current_rate))
            end

            if gen % 50 == 0
                for _ in 1:4
                    new_pop[end] = create_gene()
                end
            end

            populations[obj] = new_pop
        end

        serialize(checkpointFile, Dict(
            :populations => populations,
            :gen => gen,
            :best_by_obj => best_by_obj
        ))


        print("\rGen $gen | $(join(status_updates, " | "))    ")
        flush(stdout)
    end
end

main()
