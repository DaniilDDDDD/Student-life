import numpy as np
import random
import scipy
import scipy.linalg

def LU_decomposition(A, find_permutation=False):
    n = len(A)
    permutation = False
    C = np.array(A)
    P = np.eye(n)
    
    for i in range(n):
        pivotValue = 0
        pivot = -1
        for row in range(i,n):
            if (abs(C[row][i]) > pivotValue):
                pivotValue = abs(C[row][i])
                pivot = row
        
        if (pivotValue != 0):
            if pivot != i:
                P[[pivot, i]] = P[[i, pivot]]
                C[[pivot, i]] = C[[i, pivot]]
                permutation = not permutation
            for j in range(i+1,n):
                C[j][i] /= C[i][i]
                for k in range(i+1,n):
                    C[j][k] -= C[j][i] * C[i][k]
    
    # теперь матрица C = L + U - E
    
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    for i in range(n):
        for j in range(i):
            L[i][j] = C[i][j]
        L[i][i] = 1
        U[i][i] = C[i][i]
        for j in range(i+1,n):
            U[i][j] = C[i][j]
            
    if find_permutation:
        return L, U, P, permutation
    return L, U, P
        
    
def find_rank(A):
    n = len(A)
    C = np.array(A)
    rank = 0
    rank_found = False
    k = n - 1 # индекс последнего столбика, к которым можно поменять нулевой
    
    for i in range(n):
        pivotValue = 0
        pivot = -1
        while ((np.all(C[:,i]==0)) and (i<k)):
            C[:,[i, k]] = C[:,[k, i]]
            k -= 1
            
        for row in range(i,n): # выбираем ведущий элемент по столбцу
            if (abs(C[row][i]) > pivotValue):
                pivotValue = abs(C[row][i])
                pivot = row
        
        if (pivotValue != 0):
            if pivot != i:
                C[[pivot, i]] = C[[i, pivot]]
            for j in range(i+1,n):
                C[j][i] /= C[i][i]
                for k in range(i+1,n):
                    C[j][k] -= C[j][i] * C[i][k]
        else:
            rank -= 1
        rank += 1
        
    return rank


def solve_LU(L,U,P,b):
    """LUx=b => 1) Ly=b 2)Ux=y"""
    
    b = np.dot(P,b)
    
    y = [b[0]]
    for i in range(1,len(L)):
        cur = b[i]
        for j in range(i):
            minus = 0
            minus += L[i][j]*y[j]
            cur -= minus
        y.append(cur)
        
    x = [0]*len(U)
    x[len(U)-1] = (y[len(U)-1] / U[len(U)-1][len(U)-1])
    for i in reversed(range(len(U)-1)):
        cur = y[i]
        for j in range(i+1, len(U)):
            minus = 0
            minus += U[i][j]*x[j]
            cur -= minus
        x[i] = (cur/U[i][i])

    return np.array(x)
    
def inverse_matrix(L,U,P):
    E = np.eye(len(U))
    
    for i in range(len(U)):
        x = np.array(solve_LU(L,U,P,E[:,i]))
        E[:,i] = x.transpose()
    
    return E


def determinant(U, permutation):

    res = 1
    for i in range(len(U)):
        res *= U[i][i]
        
    if permutation:
        res*=(-1)
        return res
    return res


def main():
    #n = random.randint(0,100)
    n=5
    A = (-(2*n))*np.random.sample((n,n))+n
    B = (-(2*n))*np.random.sample((n,1))+n
    print('A: ')
    print(A)
    
    for i in range(1,n+1):
        if np.linalg.det(A[0:i, 0:i]) == 0:
            print('LU разложение невозможно!')
            break
      
    L, U, P, permutation = LU_decomposition(A, True)
    P1, L1, U1 = scipy.linalg.lu(A)
    
    print('P: ')
    print(P)
    print()
    print('L: ')
    print(L)
    print()
    print('U: ')
    print(U)
    print()
    
    print('PA и LU, соответственно:')
    print()
    print(np.dot(P,A))
    print()
    print(np.dot(L,U))
    print()
    
    print('Определитель через встроенную функцию и через написанную выше, соответственно')
    print(f'NumPy determinant A = {np.linalg.det(A)}')
    print()
    print(f'Determinant A = {determinant(U, permutation)}')
    print()
    
    print('Решение СЛАУ через scipy.linalg.solve() и через LU разложение соответственно')
    print(scipy.linalg.solve(A,B))
    print()
    x = solve_LU(L,U,P,B)
    print(x)
    print()
    print('Проверка Ax-b=0')
    print()
    print(np.dot(A,x)-B)
    print()
    
    print('Обратная матрица:')
    print()
    print(inverse_matrix(L,U,P))
    print()
    print(np.linalg.inv(A))
    print()
    print(f'Ранг матрицы через NumPy: {np.linalg.matrix_rank(A)}')
    print()
    print(f'Ранг матрицы через написанную функцию: {find_rank(A)}')
    print()
    
    D = np.array([
        [1,2,3,5,3],
        [2,6,7,7,8],
        [3,7,8,2,3],
        [0,0,0,0,0],
        [2,6,5,6,7]
    ])
# Можно генерировать рангдомную матрицу, но это очень долго
#     D  = (-(2*n))*np.random.sample((n,n))+n
#     while np.linalg.det(D)!=0:
#         D  = (-(2*n))*np.random.sample((n,n))+n
    
    print(f'Вырожденная матрица: определитель равен {np.linalg.det(D)}')
    print()
    print(D)
    print()
    print(f'Ранг матрицы через NumPy: {np.linalg.matrix_rank(D)}')
    print()
    print(f'Ранг матрицы через написанную функцию: {find_rank(D)}')
    print()

    
if __name__=='__main__':
    main()