def rna_symbol(n):
    if n == 0:
        return 'A'
    if n == 1:
        return 'C'
    if n == 2:
        return 'G'
    if n == 3:
        return 'U'
    if n >=4 :
        return 'X'

def digits_to_rna_sequence(n):
    n=n-1
    if n < 4:
        return rna_symbol(n)
    else:
        return digits_to_rna_sequence(n // 4) + rna_symbol(n % 4)



if __name__ == "__main__":
    for x in range(1,10):
        print(digits_to_rna_sequence(x))