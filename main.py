import numpy as np
import random

def func(a, b, c, x):  # Calculating value of function at x
    return c + np.add(np.dot(b.T, x), np.dot(np.dot(x.T, a), x))

def get_func_coeff(dim):
    print('Enter numbers for vector b separated by space')
    entries = list(map(int, input().split()))
    b = np.array(entries)
    while b.size != dim:
        print(f'Size of vector b must be equal to {dim}. Enter numbers for vector b separated by space')
        entries = list(map(int, input().split()))
        b = np.array(entries)

    print('Enter numbers for matrix A in order of each row separated by space')
    entries = list(map(int, input().split()))
    while True:
        if len(entries) != dim * dim:
            print(f'Matrix size must be equal to {dim}x{dim}. Enter {dim * dim} numbers')
            entries = list(map(int, input().split()))
            continue
        a = np.array(entries).reshape(b.size, b.size)
        if np.any(np.linalg.eigvals(a) <= 0): # Checking if given matrix is positive-definite
            print(f'Matrix must be positive-definite. Enter {dim * dim} numbers')
            entries = list(map(int, input().split()))
            continue
        else:
            break

    print('Enter c')
    c = float(input())
    return a, b, c

def pop_init(d,dimensions,population_size):
    pop_d = []
    pop_b = []
    for i in range(population_size):
        for j in range(dimensions):
            temp = np.random.randint(j - 2 ** d, 2 ** d)
            pop_d.append(temp)
            pop_b.append(to_gray_mod(temp, -(j - 2 ** d)-1, '0' + str((2 ** d) // 2)))
    return pop_b,pop_d

def to_gray_mod(num_i,shift,bit_len):
    num_i = num_i +shift
    num_i = num_i ^(num_i>>1)  # convert decimal into gray equivalent decimal
    x_b = format(num_i, bit_len+'b')   # convert decimal to binary
    return x_b

def to_gray(matrix,d,dimensions,population_size):
    pop_b = []
    for i in range(population_size):
        for j in range(dimensions):
            pop_b.append(to_gray_mod(matrix[i][j], -(j - 2 ** d) - 1, '0' + str((2 ** d) // 2)))
    pop_b = np.array(pop_b).reshape(population_size,dimensions)
    return pop_b


def inverse_gray_mod(x_b,shift): # from shifted gray code to decimal
    num_i = int(x_b,2)
    inv = 0
    while(num_i):
        inv = inv ^ num_i
        num_i = num_i >> 1
    return inv - shift
def reverse_gray(matrix,d,dimensions,population_size):
    pop_d = []
    for i in range(population_size):
        for j in range(dimensions):
            pop_d.append(inverse_gray_mod(matrix[i][j], -(j - 2 ** d)-1))
    pop_d = np.array(pop_d).reshape(population_size,dimensions)
    return pop_d

def score(pop_matrix,A,b,c): #Argument is matrix with size: population x dimensions
    score = []
    for i in range (len(pop_matrix)):
        score.append(float(func(A,b,c,pop_matrix[i,:])))
    return score


def roulette_wheel(pop_matrix,A,b,c): # Argument is matrix with size : population x dimensions

    score_ar = np.array(score(pop_matrix,A,b,c))
    score_ar = (score_ar - score_ar.min())/(score_ar.max() - score_ar.min())

    prob_ar = (score_ar)/(score_ar.sum())
    i = np.random.choice(pop_matrix.shape[0], size=population_size, p=prob_ar)
    return pop_matrix[i]

def crossover(prob,p1,p2):
    n = len(p1)
    m = len(p1[0])
    c1 = p1
    c2 =p2

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

def mutation(indiv,prob):
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


#Numbers for testing #Main testing
dimensions = 3 # n
d = 3
A = [[-2, 1, 0], [1, -2, 1], [0, 1, -2]]
b = [-14, 14, -2]
c = -23.5
population_size = 100
crossover_prob = 0.9
mutation_prob = 0.05
iterations = 1000
A = np.array(A).reshape(3,3)
b= np.array(b)


[pop_b,pop_d] = pop_init(d,dimensions,population_size)
pop_matrix = np.array(pop_d).reshape(population_size,dimensions)
pop_matrix_b = np.array(pop_b).reshape(population_size,dimensions)


# Not sure
for count in range(iterations):
    pop_matrix = roulette_wheel(pop_matrix,A,b,c)
    pop_matrix_b = to_gray(pop_matrix,d,dimensions,population_size)
    children = np.empty((1,1),str)
    for i in range (0,population_size,2):
        for child in crossover(crossover_prob, pop_matrix_b[i], pop_matrix_b[i + 1]):
            child = mutation(child,mutation_prob)
            children = np.append(children,child)
    children = np.delete(children,0)
    pop_matrix_b = np.array(children).reshape(population_size,dimensions)
    pop_matrix = reverse_gray(pop_matrix_b,d,dimensions,population_size)

print(pop_matrix)
temp = np.array(score(pop_matrix,A,b,c))
print(temp.max())

# final = dict(zip(pop_matrix,temp.T))
# print(final)
# matA = [[-7,-6,-5],[8,9,10]]
# temp = to_gray(matA,3,3,2)
# print(temp)
# print()
# print(reverse_gray(temp,3,3,2))
#
# print('\n\n\n')
# # num = to_gray_mod(-6,7,'08')
# # print(num)
#
# print(inverse_gray_mod('0111',7))
# print(inverse_gray_mod('0000',6))
# print(inverse_gray_mod('0001',5))
# print(inverse_gray_mod('1000',7))
# print(inverse_gray_mod('1000',6))
# print(inverse_gray_mod('1000',5))


# print(to_gray_mod(-2,7,'04'))
# print(inverse_gray_mod('0111',7))



# print(pop_matrix)
# print(score(pop_matrix,A,b,c))

# pop_matrix = [[1,1,1],[2,2,2],[3,3,3]]
# pop_matrix = np.array(pop_matrix).reshape(3,3)
# prob_ar = [0.05,0.1,0.85]
#
# i = np.random.choice(pop_matrix.shape[0], size=3, p=prob_ar)
# print(i)
# print(pop_matrix[i])
# print(roulette_wheel(pop_matrix,A,b,c))

# print(pop_matrix_b[0])
# print(pop_matrix_b[1])

# print(pop_matrix_b[1][1][:2])
# print(crossover(pop_matrix_b[0],pop_matrix_b[1],crossover_prob))

# indiv = pop_matrix_b[0]

# print(mutation(indiv,1))

# vfunc = np.vectorize(to_gray_mod)
# print(vfunc([[1,2,3],[4,5,6]],5,'8'))
# y = [1,2,3,4]
# probA = [0.1,0.2,0.3,0.4]
# indices = np.random.choice(y, size=4, p=probA)
# print(indices)
# # print(indices)

# print(pop_matrix[indices])

# test = np.array(roulette_wheel(pop_matrix,A,b,c))
# print(test)

#pop_vec = np.array(pop).reshape(population_size,dimensions)
#print(pop_vec)
#pop_array = np.array(pop)
# pop_array = np.array([0,1,2,-1])
# bin_nums = ((pop_array.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int)
# #print(bin_nums)
# final = bin_nums[:,::-1]
# print(final)




# num =7
# num = num^(num>>1)
#
# x = format(num,'08b')
# print(x)

# print(to_gray_mod(10,7,'08'))
# print(inverse_gray_mod(to_gray_mod(10,7,'08'),7))