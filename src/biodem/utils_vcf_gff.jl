using JLD2
using JSON
using GFF3
using GeneticVariation


"""
Read the GFF file and build a dictionary of gene information.
"""
function build_gene_dict(path_gff::String, save_as_jld2::Bool=false)
    reader = open(GFF3.Reader, path_gff)
    gene_dict = Dict{String,Dict}()
    for record in reader
        if GFF3.featuretype(record) == "gene"
            gene_id = GFF3.attributes(record)[1][2][1]
            chrx = GFF3.seqid(record)
            if startswith(chrx, "Chr")
                chrx = "chr" * chrx[4:end]
            end
            gene_dict[gene_id] = Dict{String,Any}(
                "seqid" => chrx,
                "start" => GFF3.seqstart(record),
                "end" => GFF3.seqend(record),
                "strand" => GFF3.strand(record),
                # "name" => GFF3.attributes(record)[3][2][1],
                # "note" => GFF3.attributes(record)[2][2][1],
            )
        end
    end
    close(reader)
    gene_ids = collect(keys(gene_dict))
    
    if save_as_jld2
        path_save = joinpath(dirname(path_gff), basename(path_gff) * ".jld2")
        @save path_save {compress=true} gene_dict gene_ids
        println("Gene dictionary saved as $path_save")
    end

    return gene_dict, gene_ids
end


"""
Read the VCF file (and GFF file if provided) and build dictionaries of SNP information and gene information.
"""
function build_snp_dict(path_vcf::String, path_gff::String="", save_jld2::Bool=true, save_json::Bool=true)
    if !isempty(path_gff)
        if endswith(path_gff, ".jld2")
            @load path_gff gff_gene_dict gff_gene_ids
        else
            gff_gene_dict, gff_gene_ids = build_gene_dict(path_gff)
        end
    end

    io = open(path_vcf, "r")
    reader = VCF.Reader(io)

    header_vcf = VCF.header(reader)
    sample_ids = header_vcf.sampleID
    n_samples = length(sample_ids)

    snp_dict = Dict{String,Dict}()
    
    snps_genes = []

    for record in reader
        chrom = VCF.chrom(record)
        if !startswith(chrom, "chr")
            chrom = "chr" * chrom
        end
        pos = VCF.pos(record)
        ref_allele = VCF.ref(record)
        alt_alleles = VCF.alt(record)
        genotypes = VCF.genotype(record, 1:n_samples, "GT")
        # snp_id = "$chrom-$pos-$ref_allele-$alt_allele"
        snp_id = "$chrom-$pos"

        if isempty(path_gff)
            # Add SNP to the dictionary
            snp_dict[snp_id] = Dict{String,Any}(
                "chrom" => chrom,
                "pos" => pos,
                "ref_allele" => ref_allele,
                "alt_alleles" => alt_alleles,
                "genotypes" => genotypes,
            )
        else
            # Check if the SNP is within a gene
            is_within_gene, gene_ids_x = check_pos(chrom, pos, gff_gene_ids, gff_gene_dict)

            if is_within_gene

                # Add gene_ids_x to the list of genes
                push!(snps_genes, gene_ids_x)

                # Add SNP to the dictionary if it is within a gene and on the positive strand
                snp_dict[snp_id] = Dict{String,Any}(
                    "chrom" => chrom,
                    "pos" => pos,
                    "ref_allele" => ref_allele,
                    "alt_alleles" => alt_alleles,
                    "genotypes" => genotypes,
                    "gene_ids" => gene_ids_x,
                )
            end
        end
    end

    close(reader)
    close(io)

    # Get the list of SNPs
    snp_ids = collect(keys(snp_dict))
    sort!(snp_ids)
    println("Number of SNPs: $(length(snp_ids))")

    # Get the unique gene names
    gene_ids_unique = unique(vcat(snps_genes...))
    sort!(gene_ids_unique)
    println("Number of unique genes: $(length(gene_ids_unique))")

    if length(gene_ids_unique) > 0
        gene_list = []
        for i in eachindex(snp_ids)
            push!(gene_list, snp_dict[snp_ids[i]]["gene_ids"])
        end
    end

    if save_jld2
        path_jld2 = joinpath(dirname(path_vcf), basename(path_vcf) * ".snp_and_gene4snp.jld2")
        @save path_jld2 {compress=true} snp_dict snp_ids sample_ids gene_list gene_ids_unique gff_gene_dict gff_gene_ids
        println("SNP and GFF information dictionary saved as $path_jld2")
    end

    if save_json
        path_json = joinpath(dirname(path_vcf), basename(path_vcf) * ".gene4snp.json")
        json_dict = Dict{String,Any}(
            "gene_list" => gene_list,
            "gene_ids" => gene_ids_unique,
            "snp_ids" => snp_ids,
        )
        open(path_json, "w") do fio
            JSON.print(fio, json_dict, 4)
        end
        println("SNP and GFF information dictionary saved as $path_json")
    end

    if save_jld2 || save_json
        return nothing
    else
        return snp_dict, snp_ids, sample_ids, gene_list, gene_ids_unique, gff_gene_dict, gff_gene_ids
    end
end


"""
Check if a SNP is within a gene.
- If the SNP is within a gene, return true and the gene information.
- Otherwise, return false and an empty String.
"""
function check_pos(chrom::String, pos::Int, gene_ids::Vector{String}, gene_dict::Dict{String,Dict})::Tuple{Bool,Vector{String}}
    # Generate a boolean vector with length length(gene_ids) to check if the SNP is within a gene
    bool_checks::Vector{Bool} = [false for i in 1:length(gene_ids)]
    Threads.@threads for i in eachindex(gene_ids)
        bool_checks[i] = check_pos(chrom, pos, gene_dict[gene_ids[i]])
    end
    if any(bool_checks)
        return true, gene_ids[findall(bool_checks)]
    else
        return false, [""]
    end
    # return any(bool_checks)
end

function check_pos(chrom::String, pos::Int, gene_dict_x::Dict)::Bool
    if gene_dict_x["seqid"] == chrom
        if gene_dict_x["start"] <= pos <= gene_dict_x["end"]
            if gene_dict_x["strand"] == GFF3.GenomicFeatures.STRAND_POS
                return true
            end
        end
    end
    return false
end
