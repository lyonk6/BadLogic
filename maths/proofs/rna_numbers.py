import math

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

def number_conversion(n):
    n=n-1
    if n < 4:
        return rna_symbol(n)
    else:
        next = n // 4
        return number_conversion(next) + rna_symbol(n - next)
