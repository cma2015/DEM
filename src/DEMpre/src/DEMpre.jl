__precompile__(true)

module DEMpre

include("modules/utils.jl")
using .MyUtils
export rename_files, my_count_lines, my_read_table, my_readline, my_write_table

include("modules/module_MyFilters.jl")
using .MyFilters
export filter_cols, filter_rows, filter_rows_SNP, filter_ath

include("modules/module_SplitData.jl")
using .SplitData
export write_randomRepeats, my_split_bootstrap, pipe_randomSplits

include("modules/module_SearchSNP.jl")
using .SearchSNP
export search_gene_of_SNP


include("modules/module_DimsReduction.jl")
using .DimsReduction
export train4embed, transform_snp, get_loader_direct, pick_SNPsInGene
#export myStructDataIn, myStructParamTrain


include("modules/module_InterpretModel.jl")
using .InterpretModel
export interpreter_inDir, interpreter

#=
function julia_main()::Cint
    # do something based on ARGS?

    if length(ARGS) < 5
        throw(ErrorException("Not enough arguments"))
    end
    file_region2snp = ARGS[1]
    file_snp_trn = ARGS[2]
    file_snp_val = ARGS[3]
    file_pheno_trn = ARGS[4]
    file_pheno_val = ARGS[5]
    train4embed(file_region2snp, file_snp_trn, file_snp_val, file_pheno_trn, file_pheno_val)

    if length(ARGS) > 5
        file_model = ARGS[6]
        # dir_models = ARGS[7]
        interpreter(file_model)
    end

    return 0 # if things finished successfully
end
=#


end # module DEMpre


using .DEMpre
