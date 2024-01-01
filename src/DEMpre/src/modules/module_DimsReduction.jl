# Dimensionality Reduction for SNP
module DimsReduction

using Flux, CUDA, BSON, Dates, Tables
include("utils_dl.jl")
include("models.jl")
if !isdefined(@__MODULE__, :MyUtils)
    include("utils.jl")
    using .MyUtils
end

export train4embed, transform_snp
export pick_SNPsInGene
export DimReduction
export run_DR
export get_loader_direct


@doc """
## Pick SNPs with gene found.
"""
function pick_SNPsInGene_ut(L_sta::Int64, L_end::Int64,
                            FileSNP::String, snp_remain::Vector{Int64}, IdPos::Vector{Int64}, w_name::String)
    #write 1st row
    @views tmpLine = my_readline(FileSNP, snp_remain[L_sta], delim=',')[Not([1,2,3,4])][IdPos]
    my_write_table(tmpLine, w_name, toTable=true)
    @views for lnN in (L_sta + 1):L_end
        tmpLine = my_readline(FileSNP, snp_remain[lnN], delim=',')[Not([1,2,3,4])][IdPos]
        my_write_table(tmpLine, w_name, isAppend=true, toTable=true)
    end
    return nothing
end


@doc """
This function takes in 3 files and returns a list of SNPs in a genomic region.
Multi-thread design for high-performance and shorter duration.
Inputs:
    FileGeneToSNP: a file containing a list of gene-SNP relations.
    FileSNP: a file containing a list of SNPs.
    FileIdWhLine: a file containing a list of IDs and positions.
"""
function pick_SNPsInGene(FileGeneToSNP::String, FileSNP::String, FileIdWhLine::String)
    IdPos = parse.(Int64, my_readline(FileIdWhLine, 1, delim='\t'))
    g2s = my_read_table(FileGeneToSNP, String, '\t')
    # Remove duplicates and sort the list of SNPs
    @views snp_remain = parse.(Int64, unique(g2s[:, 1])) |> sort
    println("\n--- SNPs with gene found: ", length(snp_remain), "\n")
    ## Multi-thread for high-performance and less duration
    # Get the number of available threads
    threadNum = Threads.nthreads()
    # Split the list of SNPs into parts for each thread
    startPts, endPts = multiThreadSplit(length(snp_remain), threadNum)
    # Iterate through each thread
    Threads.@threads for nT in 1:threadNum
        # Call the function to pick SNPs in a gene
        pick_SNPsInGene_ut(startPts[nT], endPts[nT], FileSNP, snp_remain, IdPos,
                           string(dirname(FileSNP), "/", ".tmp_nt", format_numLen(nT, 2), "_", basename(FileSNP), ".txt"))
    end
    # Concatenate the parts of the files
    conc_fileParts(string(dirname(FileSNP), "/.tmp_nt*"),
                   string(dirname(FileSNP), "/", "_lastest_", basename(FileSNP), ".txt"))
    return nothing
end

@doc """
Count GeneToSNP
"""
function count_gene2snp(fpath::String, returnWhLine::Bool = false)
    fin = my_read_table(fpath, String, '\t')
    ## Must sort SNP by which line because dataset is sorted. !!!!!
    @views orderL = sortperm(parse.(Int64, fin[:,1]))
    @views fin = fin[orderL, :]
    ##
    @views gene_uniq = unique(fin[:,2])
    numsSNP = Int64[]
    positions = []
    @views for gn in eachindex(gene_uniq)
        pos = findall(x -> x == gene_uniq[gn], fin[:,2])
        push!(numsSNP, length(pos))
        push!(positions, pos)
    end
    if returnWhLine; return positions, size(fin)[1]; end;
    return numsSNP
end


@doc """
Transform the SNP matrix to gene matrix for dimension reduction.
It is also benefitful for the downstream analysis.
    e.g. It can accelerate the training of the DEM model.
"""
function whGradsToDel(pathGeneToSNP::String)
    #Count the number of SNPs associated with each gene
    geneL, num_snp = count_gene2snp(pathGeneToSNP, true)
    #Create a matrix of weights for each SNP
    wh2Keep = zeros(Float32, length(geneL), num_snp)
    # Loop through each gene and set the weight for each SNP to 1.0
    Threads.@threads for gn in eachindex(geneL)
        wh2Keep[gn, geneL[gn]] .= 1.0
    end
    # Return the matrix of weights
    return wh2Keep
end

# =======================================

function DimReduction(LoadData::myStructLoader=myStructLoader("pathToPieces", "01", 32, "SNP", "_pheno_zscore", false, false, std_zscore!, false, false, [0,0]), 
                      paramTrain::myStructParamTrain = myStructParamTrain(550, 1e-4, 70), calc_whLayer::Int64 = 1,
                      path_Gene2SNP::String = "none", isSimpleMd::Bool=true, subDir::String="", md_dimL2::Int64=2000, isZipModel::Bool=true,
                      bias::Bool=false)
    #
    if subDir == ""; subDir = string("dr_", md_dimL2, "/"); end
    w_pref = string(LoadData.dirIn, "/", subDir, LoadData.whichPiece, "_")
    mkpath(dirname(w_pref))
    path_wrt_r = string(w_pref, "drResult_", LoadData.grepX_phase, ".txt")
    path_save_model = string(w_pref, "drModel_", LoadData.grepX_phase, ".bson")
    marks = [Dates.now(Dates.UTC), LoadData.dirIn, LoadData.whichPiece, LoadData.batchSize, lr]
    my_write_table(marks, path_wrt_r, toTable=true)
    ##
    ld1 = LoadData
    ld1.returnMatrix = true
    mx_trn, mx_val, mx_tst = get_loader(ld1)
    num_feat = size(mx_tst[1])[1]
    num_trait = size(mx_tst[2])[1]
    ##
    isFreezeMdParam = false
    where2Freeze = zeros(Float32, 2, 2)
    if LoadData.grepX_phase == "SNP"
        isFreezeMdParam = true
        model = md_snpS(path_Gene2SNP, num_trait, isSimpleMd)
        where2Freeze = whGradsToDel(path_Gene2SNP)
    else
        model = md_fc(num_feat, num_trait, md_dimL2, isSimpleMd, bias)
    end
    println(model)
    println("\n")
    #
    trn_loader, val_loader, tst_loader = get_loader(LoadData)
    dataIn = myStructDataIn(trn_loader, tst_loader, val_loader)
    trn_loader, val_loader, tst_loader = nothing, nothing, nothing
    model = myTrain(model, dataIn, paramTrain, isFreezeMdParam, true, true,
                    path_w_rec=path_wrt_r, path_save_model=path_save_model, isZipModel=isZipModel, wh2freeze=where2Freeze)
    where2Freeze = nothing
    #
    io = open(path_wrt_r, "a")
    write(io, string("\n", Dates.now(Dates.UTC), "\n\n", model, "\n"))
    close(io)
    @views begin
        g_trn = transform_snp(model, mx_trn[1], calc_whLayer, bias) |> transpose |> Tables.table ; my_write_table(g_trn, string(w_pref, "trn_dr_", LoadData.grepX_phase, ".txt")) ; g_trn = nothing;
        g_val = transform_snp(model, mx_val[1], calc_whLayer, bias) |> transpose |> Tables.table ; my_write_table(g_val, string(w_pref, "val_dr_", LoadData.grepX_phase, ".txt")) ; g_val = nothing;
        g_tst = transform_snp(model, mx_tst[1], calc_whLayer, bias) |> transpose |> Tables.table ; my_write_table(g_tst, string(w_pref, "tst_dr_", LoadData.grepX_phase, ".txt")) ; g_tst = nothing;
    end
    return nothing
end


function run_DR(whichX::Vector{String}, whichY::String, traitsIn::Vector{String}, nameSplits::Vector{String}, dirDataset::String,
                batch_size::Int64, paramTrain::myStructParamTrain, whLayerRD::Int64, MdDimLayer2::Int64=2000;
                pathG2S::String="none", suffixTrait::String="_AllOmics", subDir::String="/r10", isSimpleMd::Bool=true, isZipModel::Bool=true,
                bias::Bool=true, selectY::Bool=false, ySelected::Vector{Int64}=[0, 0], stdX::Bool=false, stdY::Bool=false, stdFunc::Function=std_zscore!)
    for wx in eachindex(whichX)
        for tnx in eachindex(traitsIn)
            for wp in nameSplits
                println("\n", "---- Omic: ", whichX[wx], "\n", "---- Trait: ", traitsIn[tnx], "\n", "---- Random split: ", wp, "\n")
                DimReduction(myStructLoader(string(dirDataset, traitsIn[tnx], suffixTrait, subDir), wp, batch_size, whichX[wx], whichY, stdX, stdY, stdFunc, false, selectY, ySelected),
                             paramTrain, whLayerRD,
                             pathG2S, isSimpleMd, "", MdDimLayer2, isZipModel, bias)
            end
        end
    end
    return nothing
end


@doc """
This function is used to perform direct dimension reduction on a given path to a SNP.
Inputs:
    path_region_to_snp: the path to the genomic regions to SNPs.
    path_x_trn & path_x_val: the paths to the training and validation sets for the SNP.
    path_y_trn & path_y_val: the paths to the training and validation sets for the trait's phenotypes.
    dir_save: the path to save the model and the training and validation SNP values.
    es_tries: the number of tries for early stopping.
    lr: learning rate.
    compress_model: a boolean for whether to compress the model for saving storage.
Outputs:
    It prints out the model, writes the model and the training and validation SNP values to a file,
    and calculates the training and validation SNP values.
"""
function train4embed(path_region_to_snp::String,
                    path_x_trn::String, path_x_val::String, path_y_trn::String, path_y_val::String;
                    dir_save::String=dirname(abspath(path_region_to_snp)),
                    batch_size::Int64=32, epoch_max::Int64=550, es_tries::Int64=70, lr::Float64=1e-4,
                    compress_model::Bool=true)#, layer_to_use::Int64=1, bias::Bool=false, std_x::Bool, std_y::Bool, std_func::Function)
    ##
    tmp_dir = tempname(dir_save, cleanup=false)
    tmp_dir = joinpath(dirname(tmp_dir), string("drSaved_", basename(tmp_dir)))
    mkpath(tmp_dir)
    println("-- Saved path: " * tmp_dir)
    path_wrt_r = joinpath(tmp_dir, "rec_snp.txt")
    path_save_model = joinpath(tmp_dir, "model_snp.bson")
    marks = [Dates.now(Dates.UTC), path_x_trn, path_y_trn, batch_size, lr]
    my_write_table(marks, path_wrt_r, toTable=true)
    ##
    mx_trn, mx_val = get_loader_direct(path_x_trn, path_y_trn, path_x_val, path_y_val, true, batch_size)
    num_trait = size(mx_trn[2])[1]
    ##
    where2Freeze = zeros(Float32, 2, 2)
    model = md_snpS(path_region_to_snp, num_trait, true)
    where2Freeze = whGradsToDel(path_region_to_snp)
    print("--      Model: ")
    println(model)
    println("\n")
    #
    dataIn = get_loader_direct(path_x_trn, path_y_trn, path_x_val, path_y_val, false, batch_size)
    paramTrain = myStructParamTrain(epoch_max, lr, es_tries)
    model = myTrain(model, dataIn, paramTrain, true, true, true,
                    path_w_rec=path_wrt_r, path_save_model=path_save_model, isZipModel=compress_model, wh2freeze=where2Freeze)
    where2Freeze = nothing
    #
    io = open(path_wrt_r, "a")
    write(io, string("\n", Dates.now(Dates.UTC), "\n\n", model, "\n"))
    close(io)
    @views begin
        g_trn = transform_snp(model, mx_trn[1], 1, false) |> transpose |> Tables.table ; my_write_table(g_trn, joinpath(tmp_dir, "trn_dr_snp.txt")) ; g_trn = nothing;
        g_val = transform_snp(model, mx_val[1], 1, false) |> transpose |> Tables.table ; my_write_table(g_val, joinpath(tmp_dir, "val_dr_snp.txt")) ; g_val = nothing;
    end
    return nothing
end


end
