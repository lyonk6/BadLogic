def digits_to_rna_sequence(n):
    n=n-1
    nucleotides = ['A', 'C', 'G', 'U']
    if n < 4:
        return nucleotides[n]
    else:
        return digits_to_rna_sequence(n // 4) + nucleotides[n % 4]

def digits_to_peptide_sequence(n):
    n=n-1
    amino_acids = [
        'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
        'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    if n < 20:
        return amino_acids[n]
    else:
        return digits_to_peptide_sequence(n // 20) + amino_acids[n % 20]

if __name__ == "__main__":
    print("RNAs:")
    for x in range(1,10):
        print(digits_to_rna_sequence(x))
    print("\nAmino acids:")
    print(digits_to_peptide_sequence(20))
    print(digits_to_peptide_sequence(40))
    print(digits_to_peptide_sequence(60))
    print(digits_to_peptide_sequence(80))
    print(digits_to_peptide_sequence(100))
    print(digits_to_peptide_sequence(200))
    print(digits_to_peptide_sequence(300))
    print(digits_to_peptide_sequence(400))
    for x in range(395, 405):
        print(digits_to_peptide_sequence(x))