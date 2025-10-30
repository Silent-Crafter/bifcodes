from Bio.Seq import Seq
from Bio.SeqUtils import gc_fraction

def find_motifs(sequence, motif):
    return [i for i in range(len(sequence)) if sequence.startswith(motif, i)]

def identify_coding_regions(sequence):
    start_codon = "ATG"
    stop_codons = {"TAA", "TAG", "TGA"}  # Using a set for faster lookup
    coding_regions = []

    i = 0
    while i < len(sequence):
        if sequence.startswith(start_codon, i):
            start_index = i
            i += 3
            while i < len(sequence):
                if sequence[i:i + 3] in stop_codons:
                    coding_regions.append((start_index, i + 3))  # Include the stop codon
                    break
                i += 3
        else:
            i += 1

    return coding_regions

dna_sequence = "ATGTCATGATATAA"
seq = Seq(dna_sequence)

gc_content = gc_fraction(seq)

motif1 = "ATA"
motif2 = "CAT"
motif1_positions = find_motifs(seq, motif1)
motif2_positions = find_motifs(seq, motif2)

print(f"Motif '{motif1}' found at positions: {motif1_positions}")
print(f"Motif '{motif2}' found at positions: {motif2_positions}")

coding_regions = identify_coding_regions(dna_sequence)
print("Coding Regions - ", coding_regions)
print("GC Content -", gc_content)


report = [
    "DNA Sequence Analysis Report\n",
    f"Provided DNA Sequence:\n{seq}\n",
    "Analysis 1: Finding Motifs\n",
    f"Motif 1 ({motif1}) found at positions: {motif1_positions}\n",
    f"Motif 2 ({motif2}) found at positions: {motif2_positions}\n",
    "Analysis 2: Calculating GC Content\n",
    f"GC Content: {gc_content:.2%}\n",
    "Analysis 3: Identifying Coding Regions\n"
]

if coding_regions:
    report.append("Coding regions found:\n")
    report.extend(f"Start: {start} Stop: {stop}\n" for start, stop in coding_regions)
else:
    report.append("No coding regions found in the sequence.")

with open("dna_sequence_analysis_report.txt", "w") as report_file:
    report_file.writelines(report)

with open("dna_sequence_analysis_report.txt", "r") as file:
    print(file.read())
