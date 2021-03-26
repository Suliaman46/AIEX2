import numpy as np
import random
import pprint


def func(a, b, c, x):  # Calculating value of function at x
    return c + np.add(np.dot(b.T, x), np.dot(np.dot(x.T, a), x))


def get_param():
    print('Enter population size:')
    pop_size = int(input())
    while pop_size <= 0:
        print('Population size must be a positive number. Enter population size:')
        pop_size = int(input())
    print('Enter crossover probability:')
    cross_prob = float(input())
    while cross_prob <= 0 or cross_prob > 1:
        print('Crossover probability must belong to (0, 1]. Enter crossover probability:')
        cross_prob = float(input())
    print('Enter mutation probability:')
    mut_prob = float(input())
    while mut_prob <= 0 or mut_prob > 1:
        print('Mutation probability must belong to (0, 1]. Enter mutation probability:')
        mut_prob = float(input())
    print('Enter number of iterations:')
    num_iter = int(input())
    while num_iter <= 0:
        print('Number of iterations must be a positive number. Enter number of iterations:')
        num_iter = int(input())
    print('Enter d for range:')
    d = int(input())
    while d < 1:
        print('Number d must be greater than 1. Enter d:')
        d = int(input())
    print('Enter number of dimensions:')
    dim = int(input())
    while dim < 1:
        print('Number of dimensions must be greater than 1. Enter number of dimensions:')
        dim = int(input())
    return pop_size, cross_prob, mut_prob, num_iter, d, dim


def get_func_coeff(dim):
    print('Enter numbers for matrix A in order of each row separated by space:')
    entries = list(map(int, input().split()))
    while len(entries) != dim * dim:
        print(f'Matrix size must be equal to {dim}x{dim}. Enter {dim * dim} numbers:')
        entries = list(map(int, input().split()))
    a = np.array(entries).reshape(dim, dim)
    print('Enter numbers for vector b separated by space:')
    entries = list(map(int, input().split()))
    b = np.array(entries)
    while b.size != dim:
        print(f'Size of vector b must be equal to {dim}. Enter numbers for vector b separated by space:')
        entries = list(map(int, input().split()))
        b = np.array(entries)
    print('Enter c')
    c = float(input())
    return a, b, c


def pop_init(d, dimensions, population_size):
    pop_d = []
    pop_b = []
    for i in range(population_size):
        for j in range(dimensions):
            temp = np.random.randint(j + 1 - 2 ** d, 2 ** d)
            pop_d.append(temp)
            pop_b.append(to_gray_mod(temp, -(j - 2 ** d) - 1, '0' + str((2 ** d) // 2)))
    return pop_b, pop_d


def to_gray_mod(num_i, shift, bit_len):
    num_i = num_i + shift
    num_i = num_i ^ (num_i >> 1)  # convert decimal into gray equivalent decimal
    x_b = format(num_i, bit_len+'b')   # convert decimal to binary
    return x_b


def to_gray(matrix, d, dimensions, population_size):
    pop_b = []
    for i in range(population_size):
        for j in range(dimensions):
            pop_b.append(to_gray_mod(matrix[i][j], -(j - 2 ** d) - 1, '0' + str((2 ** d) // 2)))
    pop_b = np.array(pop_b).reshape(population_size, dimensions)
    return pop_b


def inverse_gray_mod(x_b, shift):  # from shifted gray code to decimal
    num_i = int(x_b, 2)
    inv = 0
    while num_i:
        inv = inv ^ num_i
        num_i = num_i >> 1
    return inv - shift


def reverse_gray(matrix, d, dimensions, population_size):
    pop_d = []
    for i in range(population_size):
        for j in range(dimensions):
            pop_d.append(inverse_gray_mod(matrix[i][j], -(j - 2 ** d) - 1))
    pop_d = np.array(pop_d).reshape(population_size, dimensions)
    return pop_d


def score(pop_matrix, A, b, c):  #Argument is matrix with size: population x dimensions
    score = []
    for i in range(len(pop_matrix)):
        score.append(float(func(A, b, c, pop_matrix[i, :])))
    return score


def roulette_wheel(pop_matrix, A, b, c):  # Argument is matrix with size : population x dimensions
    score_ar = np.array(score(pop_matrix, A, b, c))
    score_ar = (score_ar - score_ar.min())/(score_ar.max() - score_ar.min())
    prob_ar = score_ar/score_ar.sum()
    i = np.random.choice(pop_matrix.shape[0], size=population_size, p=prob_ar)
    return pop_matrix[i]


def crossover(prob, p1, p2):
    n = len(p1)
    m = len(p1[0])
    c1 = p1
    c2 = p2
    temp_p1 = ''.join(p1)
    temp_p2 = ''.join(p2)
    if np.random.rand() < prob:
        point = random.randint(1, len(temp_p1) - 2)
        temp_c1 = temp_p1[:point] + temp_p2[point:]
        temp_c2 = temp_p2[:point] + temp_p1[point:]
        for i in range(0, len(temp_p1) - m + 1, m):
            np.append(c1, temp_c1[i:i + m])
            np.append(c2, temp_c2[i:i + m])
    return c1, c2


def mutation(indiv, prob):
    mut_str = []
    mutated_str = []
    m = len(indiv[0])
    for i in range(len(indiv)):
        for j in range(len(indiv[i])):
            if np.random.rand() < prob:
                mut_str.append('1') if indiv[i][j] == '0' else mut_str.append('0')
            else:
                mut_str.append(indiv[i][j])
    mut_str = ''.join(mut_str)
    for i in range(0, len(mut_str) - m + 1, m):
        mutated_str.append(mut_str[i:i + m])
    return np.array(mutated_str)


#Numbers for testing
dimensions = 3 # n
d = 3
A = [[-2, 1, 0], [1, -2, 1], [0, 1, -2]]
b = [-14, 14, -2]
c = -23.5
population_size = 50
crossover_prob = 0.9
mutation_prob = 0.05
iterations = 1000
A = np.array(A).reshape(3,3)
b = np.array(b)


#For custom data input
# population_size, crossover_prob, mutation_prob, iterations, d, dimensions = get_param()
# A, b, c = get_func_coeff(dimensions)

[pop_b,pop_d] = pop_init(d, dimensions, population_size)
pop_matrix = np.array(pop_d).reshape(population_size, dimensions)
pop_matrix_b = np.array(pop_b).reshape(population_size, dimensions)


for count in range(iterations):
    pop_matrix = roulette_wheel(pop_matrix, A, b, c)
    pop_matrix_b = to_gray(pop_matrix, d, dimensions, population_size)
    children = np.empty((1, 1), str)
    for i in range(0, population_size, 2):
        for child in crossover(crossover_prob, pop_matrix_b[i], pop_matrix_b[i + 1]):
            child = mutation(child, mutation_prob)
            children = np.append(children, child)
    children = np.delete(children, 0)
    pop_matrix_b = np.array(children).reshape(population_size, dimensions)
    pop_matrix = reverse_gray(pop_matrix_b, d, dimensions, population_size)


print(max(score(pop_matrix, A, b, c)))
final_dct = {}
for i in range (len(pop_matrix)):
    final_dct[tuple(pop_matrix[i])] = func(A,b,c,pop_matrix[i])
pprint.pprint(final_dct)


