# pregv

**`pregv`** is a fast and lightweight tool for encoding genotypes from VCF file based on genomic annotations in GFF, supporting parallel searching.

## Usage

```shell
$ pregv --help
Encode genotypes from VCF file based on GFF info.

Usage: pregv <COMMAND>

Commands:
  gff2bin  Build GFF dict
  vcf2enc  Encode genotypes from VCF file based on GFF info.
  help     Print this message or the help of the given subcommand(s)

Options:
  -h, --help     Print help
  -V, --version  Print version

```

### Step 1: Build GFF dict

```shell
$ pregv gff2bin --help
Build GFF dict

Usage: pregv gff2bin --input-gff <gff> --output <output>

Options:
  -g, --input-gff <gff>  Input GFF file.
  -o, --output <output>  Output bin.gz file.
  -h, --help             Print help
  -V, --version          Print version

```

### Step 2: Encode genotypes from VCF file

```shell
$ pregv vcf2enc --help
Encode genotypes from VCF file based on GFF info.

Usage: pregv vcf2enc [OPTIONS] --input-vcf <vcf> --input-gffdict <gffdict> --output <output>

Options:
  -v, --input-vcf <vcf>          Input VCF file
  -d, --input-gffdict <gffdict>  Input GFF dict file
  -o, --output <output>          Output pickle file
  -s, --strand <strand>          Use "+", "-" or "."(both) to specify strand [default: .]
  -m, --more-mem                 Use more RAM
  -t, --threads <nthreads>       Number of threads [default: 0]
  -h, --help                     Print help
  -V, --version                  Print version

```
