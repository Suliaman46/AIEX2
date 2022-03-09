# AIEX2
A program that finds integer numbers that maximizes a multidimensional quadratic function using the genetic algorithm.
The implemented algorithm has the following components:
* Roulette-wheel selection with scaling
* Single-point crossover
* FIFO replacement strategy
# PROBLEM DETAILS
The genetic algorithm  uses binary vectors. 

The multidimensional quadratic function for n dimensions is defined as follows:

f(x) = xT Ax + bT x + c 

where A is an n × n matrix, b is a vector of n numbers and c is a scalar.

The program allows to specify:
* the problem dimensionality,
* the range of searched integers as d ­ 1 that for each dimension i −2d ¬
xi < 2d,
* the function parameters A, b, c,
* the algorithm parameters:
 population size,
 crossover probability,
 mutation probability,
 number of algorithm iterations.
