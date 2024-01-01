module SplitData

using Random
if !isdefined(@__MODULE__, :MyUtils)
    include("utils.jl")
    using .MyUtils
end

export write_randomRepeats, my_split_bootstrap
export pipe_randomSplits


function my_split_bootstrap(num::Int64=600, seed::Int64=1234)### Size of test&validation proportion is dynamic!
    trnO = sample(MersenneTwister(seed), collect(StepRange(1, Int64(1), num)), num, replace=true, ordered=false)
    num_trn_uniq = length(unique(trnO))
    num_tstval = num - num_trn_uniq
    num_tst = ceil(num_tstval / 2) |> Int64
    tstval = setdiff(collect(StepRange(1, Int64(1), num)), unique(trnO))
    tstO = tstval[1:num_tst]
    valO = tstval[(num_tst + 1):end]
    return trnO, valO, tstO
end
function write_splits(fPaths::Vector{String}, splitN::Int64=1, seed::Int64=1234, dlm::Char='\t')
    num_sample = my_count_lines(fPaths[1])
    idx_trn, idx_val, idx_tst = my_split_bootstrap(num_sample, seed)
    MarkSplit = format_numLen(splitN, 2)
    @views for pn in eachindex(fPaths)
        fin = my_read_table(fPaths[pn], String, dlm)
        my_write_table(fin[idx_trn, :], string(dirname(fPaths[pn]), "/r10/", MarkSplit, "_trn_", basename(fPaths[pn])), toTable=true)
        my_write_table(fin[idx_val, :], string(dirname(fPaths[pn]), "/r10/", MarkSplit, "_val_", basename(fPaths[pn])), toTable=true)
        my_write_table(fin[idx_tst, :], string(dirname(fPaths[pn]), "/r10/", MarkSplit, "_tst_", basename(fPaths[pn])), toTable=true)
    end
    return nothing
end
function write_randomRepeats(fPaths::Vector{String}; repN::Int64=10, seedInit::Int64=2345, isNThreads::Bool=false, dlm::Char='\t')
    mkpath(string(dirname(fPaths[1]), "/r10/"))
    if isNThreads
        Threads.@threads for rn in 1:repN
            write_splits(fPaths, rn, (seedInit + rn), dlm)
        end
    else
        for rn in 1:repN
            write_splits(fPaths, rn, (seedInit + rn), dlm)
        end
    end
    return nothing
end

function pipe_randomSplits(dirIn::String, traits::Vector{String}, fOmics::Vector{String}, FolderSuffix::String)
    for trts in eachindex(traits)
        for oms in eachindex(fOmics)
            write_randomRepeats([string(dirIn, traits[trts], FolderSuffix, "/", fOmics[oms])], isNThreads=true)
        end
    end
    return nothing
end
#=
dirTH = "ath/dataset/THvar025/"
traits = ["FT16+RL", "CL", "RL"]
suffix = "_AllOmicsG"
omics = ["id.txt", "pheno_zscore.txt", "pheno.txt", "SNP.txt", "exp.txt", "mCG.txt", "mCHG.txt", "mCHH.txt"]
=#

end
