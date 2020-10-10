import random


# Return a random x, y Pair
def random_poly_sample(a, b, c, min , max):
    x = random.uniform(min, max)
    y = (a * x * x) + (b * x) + c

    return x, y

a = tools.random_poly_sample(2, 1, 0, -3, 3)
print("Here is a random sample: ", a)
