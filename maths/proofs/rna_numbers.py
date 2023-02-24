def digits_to_arbitrary_sequence(n, symbol_list):
    num_symbols = len(symbol_list)
    if n < num_symbols:
        return symbol_list[n]
    else:
        return digits_to_arbitrary_sequence((n // num_symbols)-1, symbol_list) + symbol_list[n % num_symbols]

def digits_to_rna_sequence(n):
    #n=n-1
    nucleotides = ['A', 'C', 'G', 'U']
    if n < 4:
        return nucleotides[n]
    else:
        return digits_to_rna_sequence((n // 4)-1) + nucleotides[n % 4]

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
    for x in range(0,10):
        print(digits_to_rna_sequence(x))

    print("\nRNAs (again):")
    for x in range(0,20):
        print(digits_to_arbitrary_sequence(x, ['A', 'C', 'G', 'U']))
    #print("\nAmino acids:")
    """
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
    """