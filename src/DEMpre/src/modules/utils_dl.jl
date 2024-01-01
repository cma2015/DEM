using CUDA, Flux, BSON, Random, StatsBase
include("types.jl")
if !isdefined(@__MODULE__, :MyUtils)
    include("utils.jl")
    using .MyUtils
end


@doc """
Mask gradient updates to weights.
"""
function zero_gradient!(myGradient, whToZeros)
    @views begin
        non_undef = filter(i -> isassigned(myGradient.grads.ht, i), 1:length(myGradient.grads.ht))## Skip "#undef"
        whCuArray = non_undef[typeof.(myGradient.grads.ht[non_undef]) .== CuArray{Float32, 2, CUDA.Mem.DeviceBuffer}]
        whCuArray = whCuArray[findall(x -> x == size(whToZeros), size.(myGradient.grads.ht[whCuArray]))]
    end
    #map!(*, myGradient.grads.ht[whCuArray[1]], whToZeros)
    map!(*, myGradient.grads.ht[whCuArray[2]], whToZeros)
end


function my_MAECor(model, dataloader, returnX::String="meanMAE")
    n_batch = 1
    size_y = size(dataloader.data[2])
    all_ŷ = zeros(Float32, size_y[1], size_y[2])
    all_y = zeros(Float32, size_y[1], size_y[2])
    for (x, y) in dataloader
        ŷ = model(gpu(x)) |> cpu
        if n_batch == 1
            all_ŷ = ŷ
            all_y = y
            n_batch = 2
        elseif n_batch > 1
            all_ŷ, all_y = hcat(all_ŷ, ŷ), hcat(all_y, y)
        end
    end
    maes, cors = zeros(Float32, size(all_y)[1]), zeros(Float32, size(all_y)[1])
    @views @inbounds @simd for bn in eachindex(maes)
        maes[bn] = Flux.mae(all_ŷ[bn, :], all_y[bn, :])
        cors[bn] = cor(all_ŷ[bn, :], all_y[bn, :])
    end
    global mean_mae = mean(maes)###############!!!!!!!!!!################
    println("     MAE: ", maes, "\n     r  : ", cors)
    if returnX == "cor"; return cors; end;
    if returnX == "MAE"; return maes; end;
    if returnX == "meanMAE"; return mean_mae; end;
    return maes, cors
end


function std_tvt(p1::Matrix, p2::Matrix, p3::Matrix, stdFunc=std_zscore!)
    conc = hcat(p1, p2, p3)
    ncol1, ncol2, ncol3 = size(p1)[2], size(p2)[2], size(p3)[2]
    stdFunc(conc)
    return conc[:, 1:ncol1], conc[:, (ncol1+1):(ncol1+ncol2)], conc[:, (ncol1+ncol2+1):end]
end

function get_loader(loadIn::myStructLoader=myStructLoader("pathToPieces", "01", 32, "SNP", "_pheno_zscore", false, false, std_zscore!, false, false, [0,0]))
    ##
    my_pwd = pwd()
    cd(loadIn.dirIn)
    ##
    grepX = string(loadIn.grepX_phase) |> Regex; grepY = string(loadIn.grepY_phase) |> Regex;
    grepTrn = string(loadIn.whichPiece, "_trn") |> Regex
    grepTst = string(loadIn.whichPiece, "_tst") |> Regex
    grepVal = string(loadIn.whichPiece, "_val") |> Regex;
    ##
    fList = readdir(loadIn.dirIn)
    fs_trn, fs_val, fs_tst = fList[occursin.(grepTrn, fList)], fList[occursin.(grepVal, fList)], fList[occursin.(grepTst, fList)]
    trn_xs, trn_ys = fs_trn[occursin.(grepX, fs_trn)], fs_trn[occursin.(grepY, fs_trn)]
    val_xs, val_ys = fs_val[occursin.(grepX, fs_val)], fs_val[occursin.(grepY, fs_val)]
    tst_xs, tst_ys = fs_tst[occursin.(grepX, fs_tst)], fs_tst[occursin.(grepY, fs_tst)]
    num_i, num_o = length(trn_xs), length(trn_ys)
    println(tst_xs, "\n", tst_ys)
    ## Ys
    trn_y, val_y, tst_y = my_read_table(trn_ys[1], Float32, '\t', true), my_read_table(val_ys[1], Float32, '\t', true), my_read_table(tst_ys[1], Float32, '\t', true)
    if loadIn.selectY; trn_y, val_y, tst_y = trn_y[loadIn.ySelected,:], val_y[loadIn.ySelected,:], tst_y[loadIn.ySelected,:] ; end;
    if num_o > 1
        for nOm in 2:num_o
            tmp_y_trn = my_read_table(trn_ys[nOm], Float32, '\t', true)
            tmp_y_val = my_read_table(val_ys[nOm], Float32, '\t', true)
            tmp_y_tst = my_read_table(tst_ys[nOm], Float32, '\t', true)
            if loadIn.selectY; tmp_y_trn, tmp_y_val, tmp_y_tst = tmp_y_trn[loadIn.ySelected,:], tmp_y_val[loadIn.ySelected,:], tmp_y_tst[loadIn.ySelected,:] ; end;
            trn_y, val_y, tst_y = vcat(trn_y, tmp_y_trn), vcat(val_y, tmp_y_val), vcat(tst_y, tmp_y_tst)
        end
        tmp_y_trn, tmp_y_val, tmp_y_tst = nothing, nothing, nothing
    end
    if loadIn.stdY
        trn_y, val_y, tst_y = std_tvt(trn_y, val_y, tst_y, loadIn.stdFunc)
    end
    ## Xs
    trn_x, val_x, tst_x = my_read_table(trn_xs[1], Float32, '\t', true), my_read_table(val_xs[1], Float32, '\t', true), my_read_table(tst_xs[1], Float32, '\t', true)
    println("Size of trn_x: ", size(trn_x))
    if num_i > 1
        for nOm in 2:num_i
            tmp_x_trn = my_read_table(trn_xs[nOm], Float32, '\t', true)
            tmp_x_val = my_read_table(val_xs[nOm], Float32, '\t', true)
            tmp_x_tst = my_read_table(tst_xs[nOm], Float32, '\t', true)
            trn_x, val_x, tst_x = vcat(trn_x, tmp_x_trn), vcat(val_x, tmp_x_val), vcat(tst_x, tmp_x_tst)
            println("Size of trn_x: ", size(trn_x))
        end
        tmp_x_trn, tmp_x_val, tmp_x_tst = nothing, nothing, nothing
    end
    if loadIn.stdX
        trn_x, val_x, tst_x = std_tvt(trn_x, val_x, tst_x, loadIn.stdFunc)
    end
    ##
    cd(my_pwd)
    ##
    trn, val, tst = (trn_x, trn_y), (val_x, val_y), (tst_x, tst_y)
    #trn_x, val_x, tst_x, trn_y, val_y, tst_y = nothing, nothing, nothing, nothing, nothing, nothing
    if loadIn.returnMatrix; return trn, val, tst ; end;
    ##
    loads = myStructDataIn(Flux.DataLoader(trn, batchsize = loadIn.batchSize, shuffle = true),
                           Flux.DataLoader(tst, batchsize = loadIn.batchSize),
                           Flux.DataLoader(val, batchsize = loadIn.batchSize, shuffle = true))
    return loads
end


function get_dataloader_tuple(loadIn::myStructLoader=myStructLoader("pathToPieces", "01", 32, "SNP", "_pheno_zscore", false, false, std_zscore!, false, false, [0,0]))
    ##
    my_pwd = pwd()
    cd(loadIn.dirIn)
    ##
    grepX = loadIn.grepX_phase |> Regex; grepY = loadIn.grepY_phase |> Regex;
    grepSplit = loadIn.whichPiece |> Regex
    grepTrn = "trn" |> Regex; grepTst = "tst" |> Regex; grepVal = "val" |> Regex;
    ##
    fList = readdir(loadIn.dirIn)
    fList = fList[occursin.(grepSplit, fList)]
    fs_trn, fs_val, fs_tst = fList[occursin.(grepTrn, fList)], fList[occursin.(grepVal, fList)], fList[occursin.(grepTst, fList)]
    trn_xs, trn_ys = fs_trn[occursin.(grepX, fs_trn)], fs_trn[occursin.(grepY, fs_trn)]
    val_xs, val_ys = fs_val[occursin.(grepX, fs_val)], fs_val[occursin.(grepY, fs_val)]
    tst_xs, tst_ys = fs_tst[occursin.(grepX, fs_tst)], fs_tst[occursin.(grepY, fs_tst)]
    num_i, num_o = length(trn_xs), length(trn_ys)
    println(tst_xs, "\n", tst_ys)
    ## Ys
    vec_y_trn = []
    vec_y_val = []
    vec_y_tst = []
    tmp_y_trn, tmp_y_val, tmp_y_tst = my_read_table(trn_ys[1], Float32, '\t', true), my_read_table(val_ys[1], Float32, '\t', true), my_read_table(tst_ys[1], Float32, '\t', true)
    if loadIn.selectY; tmp_y_trn, tmp_y_val, tmp_y_tst = tmp_y_trn[loadIn.ySelected,:], tmp_y_val[loadIn.ySelected,:], tmp_y_tst[loadIn.ySelected,:] ; end;
    if num_o > 1
        push!(vec_y_trn, tmp_y_trn)
        push!(vec_y_val, tmp_y_val)
        push!(vec_y_tst, tmp_y_tst)
        for nOm in 2:num_o
            tmp_y_trn = my_read_table(trn_ys[nOm], Float32, '\t', true)
            tmp_y_val = my_read_table(val_ys[nOm], Float32, '\t', true)
            tmp_y_tst = my_read_table(tst_ys[nOm], Float32, '\t', true)
            if loadIn.selectY; tmp_y_trn, tmp_y_val, tmp_y_tst = tmp_y_trn[loadIn.ySelected,:], tmp_y_val[loadIn.ySelected,:], tmp_y_tst[loadIn.ySelected,:] ; end;
            push!(vec_y_trn, tmp_y_trn)
            push!(vec_y_val, tmp_y_val)
            push!(vec_y_tst, tmp_y_tst)
        end
        tmp_y_trn, tmp_y_val, tmp_y_tst = nothing, nothing, nothing
    end
    ## Xs
    vec_x_trn = []
    vec_x_val = []
    vec_x_tst = []
    x_num_feat = zeros(Int64, num_i)
    tmp_x_trn, tmp_x_val, tmp_x_tst = my_read_table(trn_xs[1], Float32, '\t', true), my_read_table(val_xs[1], Float32, '\t', true), my_read_table(tst_xs[1], Float32, '\t', true)
    x_num_feat[1] = size(tmp_x_tst)[1]
    if num_i > 1
        push!(vec_x_trn, tmp_x_trn)
        push!(vec_x_val, tmp_x_val)
        push!(vec_x_tst, tmp_x_tst)
        for nOm in 2:num_i
            tmp_x_trn = my_read_table(trn_xs[nOm], Float32, '\t', true)
            tmp_x_val = my_read_table(val_xs[nOm], Float32, '\t', true)
            tmp_x_tst = my_read_table(tst_xs[nOm], Float32, '\t', true)
            push!(vec_x_trn, tmp_x_trn)
            push!(vec_x_val, tmp_x_val)
            push!(vec_x_tst, tmp_x_tst)
            x_num_feat[nOm] = size(tmp_x_tst)[1]
        end
        tmp_x_trn, tmp_x_val, tmp_x_tst = nothing, nothing, nothing
    end
    ##
    cd(my_pwd)
    ##
    if num_i > 1
        trn_x = Tuple(Matrix{Float32}(x) for x in vec_x_trn)
        val_x = Tuple(Matrix{Float32}(x) for x in vec_x_val)
        tst_x = Tuple(Matrix{Float32}(x) for x in vec_x_tst)
    else
        trn_x = tmp_x_trn
        val_x = tmp_x_val
        tst_x = tmp_x_tst
    end
    if num_o > 1
        trn_y = Tuple(Matrix{Float32}(x) for x in vec_y_trn)
        val_y = Tuple(Matrix{Float32}(x) for x in vec_y_val)
        tst_y = Tuple(Matrix{Float32}(x) for x in vec_y_tst)
    else
        trn_y = tmp_y_trn
        val_y = tmp_y_val
        tst_y = tmp_y_tst
    end
    trn, val, tst = (trn_x, trn_y), (val_x, val_y), (tst_x, tst_y)
    trn_x, val_x, tst_x, trn_y, val_y, tst_y = nothing, nothing, nothing, nothing, nothing, nothing
    if loadIn.returnMatrix; return trn, val, tst, x_num_feat ; end;
    ##
    loads = myStructDataIn(Flux.DataLoader(trn, batchsize = loadIn.batchSize, shuffle = true),
                           Flux.DataLoader(tst, batchsize = loadIn.batchSize),
                           Flux.DataLoader(val, batchsize = loadIn.batchSize, shuffle = true))
    return loads, x_num_feat
end

function get_loader_direct(path_x_trn::String, path_y_trn::String, path_x_val::String, path_y_val::String, path_x_tst::String, path_y_tst::String;
                            return_matrix::Bool=false, batch_size::Int64=16)
    ###Trash: ## At least one pair (e.g. trnX-trnY) should be filled.
    path_ifin = length.([path_x_trn, path_y_trn, path_x_val, path_y_val, path_x_tst, path_y_tst]) .> 0
    if sum(path_ifin) < 2; error("Please Input Path!"); end;
    ## Read Ys
    if path_ifin[2]; trn_y = my_read_table(path_y_trn, Float32, '\t', true); end;
    if path_ifin[4]; val_y = my_read_table(path_y_val, Float32, '\t', true); end;
    if path_ifin[6]; tst_y = my_read_table(path_y_tst, Float32, '\t', true); end;
    ## Xs
    if path_ifin[1]; trn_x = my_read_table(path_x_trn, Float32, '\t', true); end;
    if path_ifin[3]; val_x = my_read_table(path_x_val, Float32, '\t', true); end;
    if path_ifin[5]; tst_x = my_read_table(path_x_tst, Float32, '\t', true); end;
    ##
    trn, val, tst = (trn_x, trn_y), (val_x, val_y), (tst_x, tst_y)
    if return_matrix
        return trn, val, tst
    end
    ##
    loads = myStructDataIn(Flux.DataLoader(trn, batchsize = batch_size, shuffle = true),
                           Flux.DataLoader(tst, batchsize = batch_size),
                           Flux.DataLoader(val, batchsize = batch_size, shuffle = true))
    return loads
end
function get_loader_direct(path_x_trn::String, path_y_trn::String, path_x_val::String, path_y_val::String,
                            return_matrix::Bool=false, batch_size::Int64=16)
    ###Trash: ## At least one pair (e.g. trnX-trnY) should be filled.
    path_ifin = length.([path_x_trn, path_y_trn, path_x_val, path_y_val]) .> 0
    if sum(path_ifin) < 2; error("Please Input Path!"); end;
    ## Read Ys
    if path_ifin[2]; trn_y = my_read_table(path_y_trn, Float32, '\t', true); end;
    if path_ifin[4]; val_y = my_read_table(path_y_val, Float32, '\t', true); end;
    ## Xs
    if path_ifin[1]; trn_x = my_read_table(path_x_trn, Float32, '\t', true); end;
    if path_ifin[3]; val_x = my_read_table(path_x_val, Float32, '\t', true); end;
    ##
    trn, val = (trn_x, trn_y), (val_x, val_y)
    if return_matrix
        return trn, val
    end
    ##
    loads = myStructDataIn(Flux.DataLoader(trn, batchsize = batch_size, shuffle = true),
                            nothing,
                            Flux.DataLoader(val, batchsize = batch_size, shuffle = true))
    return loads
end


function save_model(model_cpu, path_save::String; isCompress::Bool=false, threadsCompress::Int64=Threads.nthreads())
    BSON.@save path_save model=model_cpu
    if isCompress
        path_xz = string(path_save, ".xz")
        rm(path_xz, force=true)
        run(`xz -9vk --threads=$threadsCompress $path_save`)
        rm(path_save)
    end
    return nothing
end


@doc """
The pipeline of training. Local/Sparse connection is supported.
(A high-efficient implementation of the pipeline, especially for sparse connection, should be considered.)
"""
function myTrain(model_cpu::Any, dataIn::myStructDataIn, paramTrain::myStructParamTrain=myStructParamTrain(3200, 1e-4, 450),
                 freeze_param::Bool=false, write_r::Bool=false, save_models::Bool=false;
                 path_w_rec::String=".dr_rslt.txt", path_save_model::String=".model.bson", isZipModel::Bool=true, wh2freeze::Matrix=zeros(Float32,1,1))
    ##
    if freeze_param; wh2freeze = wh2freeze |> gpu; end;
    model_best = model_cpu
    model = model_cpu |> gpu
    ps = Flux.params(model)
    opt = Flux.Adam(paramTrain.lr)
    loss(x, y) = Flux.Losses.mse(model(x), y)
    ## Early stopping
    es = let f = () -> my_MAECor(model, dataIn.myVal)
        Flux.early_stopping(f, paramTrain.esTries; init_score=f())
    end
    mean_mae_best = 9999999.9
    ep_best = 0
    ###============================= Train, get best epoch num ===========================###
    for ep in 1:paramTrain.epMax
        println("--- Epoch ", ep)
        for (x, y) in dataIn.myTrn
            x, y = gpu(x), gpu(y)
            gradients = gradient(() -> loss(x, y), ps)
            ###==============================================================================
            ### Set grads to zeros for freezing parameters of the layer SNP-gene!!!!!
            if freeze_param; zero_gradient!(gradients, wh2freeze); end;
            ###==============================================================================
            Flux.Optimise.update!(opt, ps, gradients)
            gradients = nothing
        end
        es() && break
        #### Save the best model to RAM
        if mean_mae_best > mean_mae
            mean_mae_best = mean_mae
            ep_best = ep
            model_best = cpu(model)
        end
    end
    model, ps, gradients, wh2freeze = nothing, nothing, nothing, nothing
    println("\n", "->> Best epoch: ", ep_best)
    if save_models; task_sMd = @task save_model(model_best, path_save_model, isCompress=isZipModel); schedule(task_sMd); end;
    ###====================================================================================###
    model = gpu(model_best)
    r_val, r_trn = my_MAECor(model, dataIn.myVal, "cor"), my_MAECor(model, dataIn.myTrn, "cor")
    if !isnothing(dataIn.myTst); r_tst = my_MAECor(model, dataIn.myTst, "cor"); end;
    #CUDA.reclaim()
    model = nothing
    if write_r
        line_tst = ""
        if !isnothing(dataIn.myTst); line_tst = string("r_tst", "\t", string.(r_tst), "\n"); end;
        io = open(path_w_rec, "a")
        write(io, string("\n\n", "ep_best", "\t", string(ep_best), "\n",
                         line_tst,
                         "r_val", "\t", string.(r_val), "\n",
                         "r_trn", "\t", string.(r_trn), "\n"))
        close(io)
    end
    if save_models; wait(task_sMd); end;
    if !isnothing(dataIn.myTst); println("\n", "  Test r: ", r_tst, "\n"); end;
    println("\n", "  Validation r: ", r_val, "\n")
    return model_best
end


@doc """
Run a given model to calculate the embeddings of the input data in a specific layer.
It runs the model on CPU.
"""
function transform_snp(model_CPU, x, whichLayer::Int64=1, bias_1st::Bool=true)
    paramt = Flux.params(model_CPU)
    out = x
    if !bias_1st
        @views out = paramt[1] * out
    else
        @views for nL in 1:whichLayer
            nLParam = 2*(nL - 1)
            out = paramt[1 + nLParam] * out
            for nc in size(out)[2]
                out[:, nc] .= (out[:, nc] + paramt[2 + nLParam])
            end
        end
    end
    return out
end
