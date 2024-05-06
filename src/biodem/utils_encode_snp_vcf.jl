# Encode SNPs using one-hot encoding.
using JLD2
using HDF5


"""
Encode a VCF file using one-hot encoding.
"""
function encode_vcf2matrix(path_vcf_jld2::String, save_as_hdf5::Bool=true)
    if endswith(path_vcf_jld2, ".jld2")
        @load path_vcf_jld2 snp_dict snp_ids sample_ids
    else
        error("File format not supported.")
    end
    sort!(snp_ids)
    n_snp = length(snp_ids)
    n_sample = length(sample_ids)
    encoded_matrix = zeros(Int8, n_snp, n_sample)

    Threads.@threads for i in 1:n_snp
        encoded_matrix[i, :] .= encode_genotype(snp_dict[snp_ids[i]]["genotypes"], snp_dict[snp_ids[i]]["ref_allele"], snp_dict[snp_ids[i]]["alt_alleles"])
    end

    if save_as_hdf5
        path_h5 = joinpath(dirname(path_vcf_jld2), "encoded_snp_matrix_" * splitext(basename(path_vcf_jld2))[1] * ".h5")
        h5write(path_h5, "encoded_matrix", encoded_matrix)
        h5write(path_h5, "snp_ids", snp_ids)
        h5write(path_h5, "sample_ids", sample_ids)
        return path_h5
    end
    
    return encoded_matrix, snp_ids, sample_ids
end

"""
This function first encodes the allele to a vector using one-hot encoding,
then convert the vector to a 16-bit integer for storage.

10 bits one-hot are used to encode the 10 combinations of base pair.

_Ten combinations are:_
- `["A/A", "C/C", "G/G", "T/T", "A/C", "A/G", "A/T", "C/G", "C/T", "G/T"]`

_Note:_
- Encoded `"A/T"` is equivalent to encoded `"T/A"` and vice versa.
"""
function encode_allele(allele::String="C|A")::Int8
    # sort the allele to make sure it is in the same order as the 10-combinations.
    if allele[1] > allele[3]
        allele_sorted = allele[3] * "/" * allele[1]
    else
        allele_sorted = allele[1] * "/" * allele[3]
    end

    # Initialize the encoded allele to a 16-length zeros vector, which also represents missing data.
    # encoded_allele = zeros(Int8, 16)
    which_hot = 0

    # Encode the allele to a 10-length vector using one-hot encoding.
    ten_combinations = ["A/A", "C/C", "G/G", "T/T", "A/C", "A/G", "A/T", "C/G", "C/T", "G/T"]

    find_hot = findfirst(x -> x == allele_sorted, ten_combinations)
    if find_hot !== nothing
        # encoded_allele[find_hot] = 1
        which_hot = find_hot
    end
    
    return which_hot
end


"""
Encode a genotype (e.g. "0|1") using one-hot encoding.
"""
function encode_genotype(genotypes::Array{String, 1}, ref_allele::String, alt_alleles::Vector{String})::Vector{Int8}
    # Build map from numeric genotype to allele string.
    genotype_map = Dict{String, String}()
    genotype_map["."] = "N"
    genotype_map["0"] = ref_allele
    for alt_a in eachindex(alt_alleles)
        genotype_map[string(alt_a)] = alt_alleles[alt_a]
    end

    encoded_alleles = zeros(Int8, length(genotypes))
    # Convert numeric genotype to allele string.
    for i in eachindex(encoded_alleles)
        # Encode the allele string using one-hot encoding.
        encoded_alleles[i] = encode_allele(genotype_map[genotypes[i][1] * ""] * "/" * genotype_map[genotypes[i][3] * ""])
    end
    return encoded_alleles
end
