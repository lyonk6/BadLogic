def digits_to_sequence(n, symbol_list):
    num_symbols = len(symbol_list)
    if n < num_symbols:
        return symbol_list[n]
    else:
        return digits_to_sequence((n // num_symbols)-1, symbol_list) + symbol_list[n % num_symbols]


if __name__ == "__main__":
    nucleotides = 'ACGU'
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    print("\nAmino acids:")
    for x in range(0, 21):
        print(digits_to_sequence(x, amino_acids))

    print("...")
    for x in range(419, 422):
        print(digits_to_sequence(x, amino_acids))

    print("RNAs:")
    for x in range(0,21):
        print(digits_to_sequence(x, nucleotides))
    print(digits_to_sequence(43, nucleotides))