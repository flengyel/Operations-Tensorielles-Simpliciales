from operations_tensorielles_simpliciales import tensor_inner_horn_rank_dimension_conjecture
from operations_tensorielles_simpliciales import tensor_inner_horn_rank_dimension_comparison
import random

def random_shape(n=14):
    length = random.randint(2, n/2)  # Length of at least two and bounded by 10
    return tuple(random.randint(2, n) for _ in range(length))  # Positive integers at least two and bounded by 10

if __name__ == "__main__":
    non_unique_horns = 0
    unique_horns = 0
    for i in range(1000):
        shape = random_shape()
        print(i ,": Shape = ", shape)
        comparison = tensor_inner_horn_rank_dimension_comparison(shape, verbose=True)
        conjecture = tensor_inner_horn_rank_dimension_conjecture(shape, verbose=True)
        if comparison != conjecture:
            print("Conjecture failed at shape = ", shape)
            exit(-1)
        if comparison:
            non_unique_horns += 1
        else:
            unique_horns += 1
    print("Conjecture verified for 1000 random shapes. Unique horns:", unique_horns, "non-unique horns:", non_unique_horns)
