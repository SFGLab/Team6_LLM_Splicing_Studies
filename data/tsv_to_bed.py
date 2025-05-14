#!/usr/bin/env python3
"""
Convert TSV file containing neojunction data to BED format, 
mapping chromosome names to NCBI accessions for GRCh37.
"""

import argparse
import sys

def main():
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description='Convert neojunction TSV to BED format with GRCh37 NCBI accessions')
    parser.add_argument('-i', '--input', required=True, help='Input TSV file containing neojunction data')
    parser.add_argument('-o', '--output', required=True, help='Output BED file path')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Define chromosome name to NCBI accession mapping for GRCh37
    GRCH37_ACCESSION_MAP = {
        "1": "NC_000001.10", "chr1": "NC_000001.10",
        "2": "NC_000002.11", "chr2": "NC_000002.11",
        "3": "NC_000003.11", "chr3": "NC_000003.11",
        "4": "NC_000004.11", "chr4": "NC_000004.11",
        "5": "NC_000005.9",  "chr5": "NC_000005.9",
        "6": "NC_000006.11", "chr6": "NC_000006.11",
        "7": "NC_000007.13", "chr7": "NC_000007.13",
        "8": "NC_000008.10", "chr8": "NC_000008.10",
        "9": "NC_000009.11", "chr9": "NC_000009.11",
        "10": "NC_000010.10", "chr10": "NC_000010.10",
        "11": "NC_000011.9", "chr11": "NC_000011.9",
        "12": "NC_000012.11", "chr12": "NC_000012.11",
        "13": "NC_000013.10", "chr13": "NC_000013.10",
        "14": "NC_000014.8", "chr14": "NC_000014.8",
        "15": "NC_000015.9", "chr15": "NC_000015.9",
        "16": "NC_000016.9", "chr16": "NC_000016.9",
        "17": "NC_000017.10", "chr17": "NC_000017.10",
        "18": "NC_000018.9", "chr18": "NC_000018.9",
        "19": "NC_000019.9", "chr19": "NC_000019.9",
        "20": "NC_000020.10", "chr20": "NC_000020.10",
        "21": "NC_000021.8", "chr21": "NC_000021.8",
        "22": "NC_000022.10", "chr22": "NC_000022.10",
        "X": "NC_000023.10", "chrX": "NC_000023.10",
        "Y": "NC_000024.9", "chrY": "NC_000024.9",
        "MT": "NC_012920.1", "chrM": "NC_012920.1", "M": "NC_012920.1"  # Mitochondrial
    }
    
    if args.verbose:
        print(f"Starting BED file creation. Input: {args.input}, Output: {args.output}")
    
    try:
        with open(args.input, 'r') as f_in, open(args.output, 'w') as f_out:
            header_line = f_in.readline().strip()
            if not header_line:
                print("Error: Input TSV file appears to be empty or header is missing.", file=sys.stderr)
                return 1
                    
            header = header_line.split('\t')
            
            try:
                gene_id_col = header.index("Gene_ID")
                coord_col = header.index("Neojunction_Coordinate")
                if args.verbose:
                    print(f"Found 'Gene_ID' at column index {gene_id_col} and 'Neojunction_Coordinate' at column index {coord_col}.")
            except ValueError:
                print(f"Error: 'Gene_ID' or 'Neojunction_Coordinate' not found in header of input file.", file=sys.stderr)
                print(f"Header was: {header}", file=sys.stderr)
                return 1
        
            processed_lines = 0
            unmapped_chroms = set()
            for line_number, line in enumerate(f_in, 1):
                parts = line.strip().split('\t')
                if len(parts) <= max(gene_id_col, coord_col):
                    continue
        
                gene_id = parts[gene_id_col]
                coord_str = parts[coord_col]
        
                chrom_parts = coord_str.split(':')
                
                original_chrom_from_coord = ""
                if len(chrom_parts) >= 3:
                    range_str = chrom_parts[-1]
                    strand = chrom_parts[-2]
                    original_chrom_from_coord = ":".join(chrom_parts[:-2])
                    
                    if strand not in ['+', '-'] or not all(c.isdigit() or c == '-' for c in range_str.replace('-', '')):
                        if len(chrom_parts) == 3:
                            original_chrom_from_coord = chrom_parts[0]
                            strand = chrom_parts[1]
                            range_str = chrom_parts[2]
                        else:
                            if args.verbose:
                                print(f"Warning: Line {line_number + 1} has unparseable coordinate string (A) '{coord_str}'. Skipping.", 
                                      file=sys.stderr)
                            continue
                else:
                    if args.verbose:
                        print(f"Warning: Line {line_number + 1} has unparseable coordinate string (B) '{coord_str}'. Skipping.", 
                              file=sys.stderr)
                    continue
                    
                try:
                    start_str, end_str = range_str.split('-')
                    start = int(start_str)
                    end = int(end_str)
                except ValueError:
                    if args.verbose:
                        print(f"Warning: Line {line_number + 1} has invalid range '{range_str}' in '{coord_str}'. Skipping.", 
                              file=sys.stderr)
                    continue
        
                if start > end:
                    start, end = end, start
                
                final_bed_chrom = GRCH37_ACCESSION_MAP.get(original_chrom_from_coord)
                
                if final_bed_chrom is None:
                    # If not found in map, it might be an unlocalized contig (like NT_...)
                    # or another name not in our simple map
                    if original_chrom_from_coord not in unmapped_chroms and args.verbose:
                        print(f"Warning: Chromosome name '{original_chrom_from_coord}' from '{coord_str}' not found in explicit map. "
                              f"Using it directly. It might be an unplaced contig or require specific mapping if errors persist for it.", 
                              file=sys.stderr)
                        unmapped_chroms.add(original_chrom_from_coord)
                    final_bed_chrom = original_chrom_from_coord  # Use original if not in map
        
                bed_start = start - 1  # BED format is 0-based
                bed_end = end
                name = f"{gene_id}|{coord_str}"
                
                f_out.write(f"{final_bed_chrom}\t{bed_start}\t{bed_end}\t{name}\t0\t{strand}\n")
                processed_lines += 1
        
        if args.verbose:
            print(f"BED file created: {args.output}. Processed {processed_lines} data lines.")
            if unmapped_chroms:
                print(f"The following chromosome names from your input were not in the standard map and were used directly: "
                      f"{sorted(list(unmapped_chroms))}")
                      
        return 0
        
    except IOError as e:
        print(f"Error accessing files: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())