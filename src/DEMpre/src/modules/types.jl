mutable struct myStructParamTrain
    epMax::Int64
    lr::Float64
    esTries::Int64
end

mutable struct myStructDataIn
    myTrn::Any
    myTst::Any
    myVal::Any
end

mutable struct myStructLoader
    dirIn::String
    whichPiece::String
    batchSize::Int64
    grepX_phase::String
    grepY_phase::String
    stdX::Bool
    stdY::Bool
    stdFunc::Function
    returnMatrix::Bool
    selectY::Bool
    ySelected::Vector{Int64}
end
