__precompile__(true)

module DEMpre

include("modules/utils.jl")
using .MyUtils
export rename_files, my_count_lines, my_read_table, my_readline, my_write_table


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


end # module DEMpre


using .DEMpre
