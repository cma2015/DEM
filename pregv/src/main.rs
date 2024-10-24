use clap::{self, Arg, ArgAction, Command};
// use rayon::prelude::*;
use pregv::{build_gff_dict, vcf2encoded};

fn main() {
    let matches = cli().get_matches();
    let subcommand = matches.subcommand();

    match subcommand {
        Some(("gff2bin", sub_matches)) => build_gff_dict(
            sub_matches
                .get_one::<String>("gff")
                .map(|s| s.as_str())
                .unwrap(),
            sub_matches
                .get_one::<String>("output")
                .map(|s| s.as_str())
                .unwrap(),
        )
        .expect("Failed to build GFF dict"),
        Some(("vcf2enc", sub_matches)) => vcf2encoded(
            sub_matches
                .get_one::<String>("vcf")
                .map(|s| s.as_str())
                .unwrap(),
            sub_matches
                .get_one::<String>("gffdict")
                .map(|s| s.as_str())
                .unwrap(),
            sub_matches
                .get_one::<String>("output")
                .map(|s| s.as_str())
                .unwrap(),
            sub_matches
                .get_one::<String>("strand")
                .map(|s| s.as_str())
                .unwrap(),
            sub_matches.get_one::<bool>("moremem").unwrap().to_owned(),
            sub_matches.get_one::<usize>("nthreads").unwrap().to_owned(),
        )
        .expect("Failed to encode genotypes"),
        _ => unreachable!("Please use available subcommands"),
    }
}

fn cli() -> Command {
    Command::new("pregv")
        .bin_name("pregv")
        .version("0.1.0")
        .author("Chenhua Wu, chanhuawu@outlook.com")
        .about("Encode genotypes from VCF file based on GFF info.")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .propagate_version(true)
        .subcommand(
            Command::new("gff2bin").about("Build GFF dict").args([
                Arg::new("gff")
                    .long("input-gff")
                    .short('g')
                    .help("Input GFF file.")
                    .required(true)
                    .action(ArgAction::Set),
                Arg::new("output")
                    .long("output")
                    .short('o')
                    .help("Output bin.gz file.")
                    .required(true)
                    .action(ArgAction::Set),
            ]),
        )
        .subcommand(
            Command::new("vcf2enc")
                .about("Encode genotypes from VCF file based on GFF info.")
                .args([
                    Arg::new("vcf")
                        .long("input-vcf")
                        .short('v')
                        .help("Input VCF file")
                        .required(true)
                        .action(ArgAction::Set),
                    Arg::new("gffdict")
                        .long("input-gffdict")
                        .short('d')
                        .help("Input GFF dict file")
                        .required(true)
                        .action(ArgAction::Set),
                    Arg::new("output")
                        .long("output")
                        .short('o')
                        .help("Output pickle file")
                        .required(true)
                        .action(ArgAction::Set),
                    Arg::new("strand")
                        .long("strand")
                        .short('s')
                        .help("Use \"+\", \"-\" or \".\"(both) to specify strand")
                        .required(false)
                        .default_value(".")
                        .action(ArgAction::Set),
                    Arg::new("moremem")
                        .long("more-mem")
                        .short('m')
                        .help("Use more RAM")
                        .required(false)
                        .action(ArgAction::SetTrue),
                    Arg::new("nthreads")
                        .value_parser(clap::value_parser!(usize))
                        .long("threads")
                        .short('t')
                        .help("Number of threads [default: 0, use all available threads]")
                        .required(false)
                        .default_value("0")
                        .action(ArgAction::Set),
                ]),
        )
}
