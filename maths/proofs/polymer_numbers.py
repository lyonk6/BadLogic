def digits_to_sequence(n, symbol_list):
    num_symbols = len(symbol_list)
    if n < num_symbols:
        return symbol_list[n]
    else:
        return digits_to_sequence((n // num_symbols)-1, symbol_list) + symbol_list[n % num_symbols]


def sequence_to_digits(sequence, symbol_map):
    if len(sequence) <= 1:
        return symbol_map[sequence]
    else:
        power = (len(symbol_map) ** (len(sequence)-1))
        power_sum = power + power * symbol_map[sequence[:1]]
        return sequence_to_digits(sequence[1:], symbol_map) + power_sum


def make_symbol_map(symbol_list):
    symbol_map = {}
    for i, x in enumerate(symbol_list):
        symbol_map[x] = i
    return symbol_map

if __name__ == "__main__":
    nucleotides = 'ACGU'
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    print("Count to 20:")
    for x in range(0, 21):
        print(sequence_to_digits(digits_to_sequence(x, amino_acids), make_symbol_map(amino_acids)))

    print("\nAmino acids:")
    for x in range(0, 21):
        print(digits_to_sequence(x, amino_acids))
    print("...")
    for x in range(419, 422):
        print(digits_to_sequence(x, amino_acids))
#
#    print("RNAs:")
#    for x in range(0,21):
#        print(digits_to_sequence(x, nucleotides))
#    print(digits_to_sequence(43, nucleotides))

    print(sequence_to_digits(nucleotides, make_symbol_map(nucleotides)))
    #print(digits_to_sequence(111, nucleotides))