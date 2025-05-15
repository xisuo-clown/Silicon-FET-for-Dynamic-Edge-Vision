def compute_for_sum(F,n,v,u):
    from math import cos
    sum_all=0
    for i in range(n):
        sum_all+=F[i][v]*cos((2*i+1)*u*math.pi/8)
    return sum_all
import math
a = [1 / 2, 1 / math.sqrt(2), 1 / math.sqrt(2), 1 / math.sqrt(2)]
dct = [[0 for _ in range(len(a))] for _ in range(len(a))]
img = [
    [10, 10, 0, 0],
    [10, 10, 0, 0],
    [0, 0, 20, 20],
    [0, 0, 20, 20],
]
F = [
    [10, 9.24, 0, -3.83],
    [10, 9.24, 0, -3.83],
    [20, -18.48, 0, 7.65],
    [20, -18.48, 0, 7.65],
]
for i in range(len(a)):
    for j in range(len(a)):
        dct[i][j] = a[i] * compute_for_sum(F,4,j,i)
print(dct)