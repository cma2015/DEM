# Simple Model Interpretability
module InterpretModel

using BSON, Flux, Random, Tables

if !isdefined(@__MODULE__, :MyUtils)
    include("utils.jl")
    using .MyUtils
end

export interpreter_inDir, interpreter


@doc """
Release the weights and bias of the model to a tsv file.
"""
function interpreter(path_model::String, bias::Bool=false)
    # Load the model from the given path
    BSON.@load path_model model
    path_wt = string(dirname(path_model), "/", "interp_weight_", basename(path_model)[1:(end-5)], ".txt")
    # Write the weight to the weight file
    my_write_table(model.layers[1].weight, path_wt, toTable=true)
    if bias
        path_bias = string(dirname(path_model), "/", "interp_bias_", basename(path_model)[1:(end-5)], ".txt")
        # Write the bias to the bias file
        my_write_table(model.layers[1].bias, path_bias, toTable=true)
    end
    return nothing
end

@doc """
Release each model's weights in a directory
"""
function interpreter_inDir(dir_models::String, bias::Bool=true)
    grepMd = ".bson" |> Regex
    fList = readdir(dir_models)
    # Find all .bson files
    mds = fList[occursin.(grepMd, fList)]
    println(mds)
    # Iterate through each .bson file in the directory
    Threads.@threads for mdn in eachindex(mds)
        # Call the interpreter function for each .bson file
        interpreter(string(dir_models, "/", mds[mdn]), bias)
    end
    return nothing
end


end # of InterpretModel
