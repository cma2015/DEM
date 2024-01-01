module SearchSNP

if !isdefined(@__MODULE__, :MyUtils)
    include("utils.jl")
    using .MyUtils
end

export search_gene_of_SNP


@doc """
Take a vector of integers and return the product of all elements in the vector
"""
function x_in_vecInt(vecIn::Vector{Int64})::Int64
    mtx = 1
    for nElem in eachindex(vecIn)
        mtx = mtx * vecIn[nElem]
    end
    return mtx
end

@doc """
Warn: This function is not stable.
"""
function BS_approx_num(goal::Int64, pathF::String, nrowF::Int64=my_count_lines(pathF), inWhichCol::Vector{Int64}=[4,5])::Int64
    len_min = maximum(inWhichCol)
    guessO::Float64 = ceil(nrowF / 2)#floor??
    guessΔ::Float64 = ceil(nrowF / 4)
    xTry = Vector{Int64}(undef, length(inWhichCol))
    xTryStr = my_readline(pathF, Int64(guessO), maxLen=50)
    #if only(xTryStr) == ""; xTryStr = my_readline(pathF, Int64(guessO)-1, maxLen=50); end;
    xTry .= parse.(Int64, xTryStr[inWhichCol])
    mark_cantfind::Int64 = 0
    while x_in_vecInt(sign.(goal .- xTry)) > 0
        if sum(goal .- xTry) > 0
            guessO += guessΔ
        elseif sum(goal .- xTry) < 0
            guessO -= guessΔ
        end
        xTryStr = my_readline(pathF, Int64(guessO), maxLen=50)
        if length(xTryStr) < len_min
            guessO -= 1
            xTryStr = my_readline(pathF, Int64(guessO), maxLen=50)
        end
        xTry .= parse.(Int64, xTryStr[inWhichCol])
        guessΔ = ceil(guessΔ / 2)
        if guessΔ < 2
            mark_cantfind += 1
            if mark_cantfind > 4 # Can't find goal
                return 0
            end
        end
    end
    return Int64(guessO)
end


function util_search_gene_by_snp(nChr::String, nPos::Int64, markL::Int64, pathGTF_chrX::String, nRowGTFx::Int64, intergenic::Bool)::String
    outVec = Vector{String}(undef, 5)
    # Binary search
    lineApprox = BS_approx_num(nPos, pathGTF_chrX, nRowGTFx, Int64[4,5])
    if lineApprox == 0
        if intergenic
            geneSelfDef = string("interG_", nChr, "_", nPos)
            outVec = string.([markL, geneSelfDef, nChr, nPos, ""])
        end
    else
        tmpO = my_readline(pathGTF_chrX, lineApprox)[[1,9]]
        gene! = split(tmpO[2], "\"")[2]
        #### | which line in SNPmx | gene | nChr | nPos | seq id |
        outVec = string.([markL, gene!, nChr, nPos, tmpO[1]])
    end
    if isassigned(outVec, 1)
        out = join(outVec, "\t")
    else
        out = ""
    end
    return out
end


##==============================================- Main utils -==============================================##

@doc """
Give simple results:
| which line in SNPmx | gene | Chr | SNP Position | seq id |
"""
function search_gene_by_snp(pathSNPmx::String, dirGTF::String, nRowGTF::Vector{Int64},
                                LnBegin::Int64, LnEnd::Int64, nT::Int64,
                                intergenic::Bool=true, skipRow::Int64=0)
    ##
    dirW = dirname(pathSNPmx) * "/"
    pathGeneOut = string(dirW, ".p-gene-", format_numLen(nT, 2), "-", basename(pathSNPmx)[1:end-4], "_", basename(dirGTF), ".txt")
    rm(pathGeneOut, force=true)
    tmpG::String = "NA"
    ##
    io = open(pathGeneOut, "a")
    for Lx in LnBegin:LnEnd
        markL = Lx - skipRow
        posPart = my_readline(pathSNPmx, Lx, delim=',', maxLen=36, nElem=2)
        if length(posPart) > 1
            nChr = posPart[1][4:end] |> String
            nPos = parse(Int64, posPart[2])
            pathGTF_chrX = string(dirGTF, "/", "AT", nChr, "G.gtf") ## The GTF file to read
            ## Search
            tmpG = util_search_gene_by_snp(nChr, nPos, markL, pathGTF_chrX, nRowGTF[parse(Int64, nChr)], intergenic)
            if length(tmpG) > 0; write(io, tmpG, "\n"); end;
        end
    end
    close(io)
    return nothing
end


@doc """
Search the gene of SNP in GTF file.
"""
function search_gene_of_SNP(pathSNPmx::String, dirGTF::String, intergenic::Bool=true; skipRow::Int64=0)
    # sh split_chr.sh
    # Count lines for GTF of each Chr
    GTFs = readdir(dirGTF)
    GTFs = GTFs[occursin.(Regex(".gtf"), GTFs)]
    nRowGTFs = Vector{Int64}(undef, length(GTFs))
    Threads.@threads for gn in eachindex(GTFs)
        nRowGTFs[gn] = my_count_lines(dirGTF * "/" * GTFs[gn])
    end
    # Run
    num_thread = Threads.nthreads()
    num_SNP = my_count_lines(pathSNPmx) - skipRow
    staL, endL = multiThreadSplit(num_SNP, num_thread)
    Threads.@threads for nT in 1:num_thread
        search_gene_by_snp(pathSNPmx, dirGTF, nRowGTFs, (skipRow + staL[nT]), (skipRow + endL[nT]), nT, intergenic, skipRow)
    end
    # Combine file parts
    dirW = dirname(pathSNPmx) * "/"
    if intergenic
        pathGeneOut = *(dirW, "_gene-", "withIntergenic_", basename(pathSNPmx)[1:end-4], "_", basename(dirGTF), ".txt")
    else
        pathGeneOut = *(dirW, "_gene-", "onlyGene_", basename(pathSNPmx)[1:end-4], "_", basename(dirGTF), ".txt")
    end
    grep_gene = dirW * ".p-gene-*"
    conc_fileParts(grep_gene, pathGeneOut)
    return nothing
end


end # of module SearchSNP
