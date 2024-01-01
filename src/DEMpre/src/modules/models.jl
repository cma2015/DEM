## Models struc for SNP transformation
using Flux

function md_snpS(FileGeneToSNP::String, num_y::Int64 = 1, isSimple::Bool = false)
    ## Get SNP foreach gene
    geneL, numSNP = count_gene2snp(FileGeneToSNP, true)
    numGene = length(geneL)
    ## Initialize weight matrix
    wt = zeros(Float32, numGene, size(my_read_table(FileGeneToSNP, String, '\t'))[1])
    for gn in eachindex(geneL)
        wt[gn, geneL[gn]] .= 1.0
    end
    ##
    if isSimple
        model = Chain(Dense(wt), Dense(numGene => num_y))
    else
        model = Chain(Dense(wt),
                    #Dropout(0.1),
                    Dense(numGene => 1000),
                    Dropout(0.1),
                    Dense(1000 => 256),
                    Dense(256 => 64),
                    Dense(64 => num_y))
    end
    return model
end


function md_fc(dimIn::Int64, dimOut::Int64, dim2use::Int64=2000, isSimpleMd::Bool=false, with_bias::Bool=false)
    if isSimpleMd
        return Chain(Dense(dimIn => dim2use, bias=with_bias), Dense(dim2use => dimOut))
    else
        return Chain(Dense(dimIn => dim2use, bias=with_bias),
                Dropout(0.2),
                Dense(dim2use => 1000),
                Dropout(0.1),
                Dense(1000 => 128),
                Dense(128 => dimOut))
    end
end
