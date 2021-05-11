import numpy as np
import random
import scipy
import scipy.linalg

# def LU_decomposition(A, find_permutation=False):
#     n = len(A)
#     permutation = False
#     C = np.array(A)
#     P = np.eye(n)
    
#     for i in range(n): 
#         pivotValue = 0
#         pivot = -1
#         for row in range(i,n): # ищем максимальный элемент в столбце
#             if (abs(C[row][i]) > pivotValue):
#                 pivotValue = abs(C[row][i])
#                 pivot = row
        
#         if (pivotValue != 0):
#             if pivot != i:
#                 P[[pivot, i]] = P[[i, pivot]]
#                 C[[pivot, i]] = C[[i, pivot]]
#                 permutation = not permutation
#             for j in range(i+1,n):
#                 C[j][i] /= C[i][i]
#                 for k in range(i+1,n):
#                     C[j][k] -= C[j][i] * C[i][k]
    
#     # теперь матрица C = L + U - E
    
#     L = np.zeros((n,n))
#     U = np.zeros((n,n))
#     for i in range(n):
#         for j in range(i):
#             L[i][j] = C[i][j]
#         L[i][i] = 1
#         U[i][i] = C[i][i]
#         for j in range(i+1,n):
#             U[i][j] = C[i][j]
            
#     if find_permutation:
#         return L, U, P, permutation
#     return L, U, P

def LU_decomposition(A, find_permutation=False):
    n = len(A)
    permutation = False
    C = np.array(A, dtype=float)
    P = np.eye(n)
    Q = np.eye(n)
    t = n - 1
    
    for i in range(n): 
        pivotValue = 0
        pivot = -1
        
        while ((np.all(C[:,i]==0)) and (i<t)):
            C[:,[i, t]] = C[:,[t, i]]
            Q[:,[i, t]] = Q[:,[t, i]]
            permutation = not permutation
            t -= 1
        
        for row in range(i,n): # ищем максимальный элемент в столбце
            if (abs(C[row][i]) > pivotValue):
                pivotValue = abs(C[row][i])
                pivot = row
        
        
        if (pivotValue != 0):
            if pivot != i:
                P[[pivot, i]] = P[[i, pivot]]
                C[[pivot, i]] = C[[i, pivot]]
                permutation = not permutation
            for j in range(i+1,n):
#                 print('in j')
#                 print(C)
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
        return L, U, P, Q, permutation
    return L, U, P, Q
        
    
def find_rank(A):
    n = len(A)
    C = np.array(A)
    rank = 0
    rank_found = False
    k = n - 1 # индекс последнего столбика, к которым можно поменнять нулевой
    
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


# def solve_LU(L,U,P,b):
#     """LUx=b => 1) Ly=b 2)Ux=y"""
    
#     b = np.dot(P,b)
    
#     y = [b[0]]
#     for i in range(1,len(L)):
#         cur = b[i]
#         for j in range(i):
#             minus = 0
#             minus += L[i][j]*y[j]
#             cur -= minus
#         y.append(cur)
        
#     x = [0]*len(U)
#     x[len(U)-1] = (y[len(U)-1] / U[len(U)-1][len(U)-1])
#     for i in reversed(range(len(U)-1)):
#         cur = y[i]
#         for j in range(i+1, len(U)):
#             minus = 0
#             minus += U[i][j]*x[j]
#             cur -= minus
#         x[i] = (cur/U[i][i])

#     return np.array(x)

def solve_LU(L,U,P,Q,b):
    """PLUx=Pb ; LU=PAQ => 1) Ly=Pb 2)Uz=y 3) x=Qz"""
    
    b = np.dot(P,b)
    
    y = [b[0]]
    for i in range(1,len(L)):
        cur = b[i]
        for j in range(i):
            minus = 0
            minus += L[i][j]*y[j]
            cur -= minus
        y.append(cur)
        
    rank = len(U)
        
    # проверка на совместность
    for i in range(len(U)):
        flag = True
        for j in U[i]:
            if abs(j) >= 1e-12:
                flag = False
                break
        if flag:
            rank -= 1
        if abs(y[i])>=10e-12 and flag:
            raise ValueError('Система несовместна')
        
    x =np.array([0]*len(U), dtype=float)
    x[rank-1] = (y[rank-1] / U[rank-1][rank-1])
    for i in reversed(range(rank-1)):
        cur = y[i]
        for j in range(i+1, rank):
            minus = 0
            minus += U[i][j]*x[j]
            cur -= minus
        x[i] = (cur/U[i][i])
        
    x = np.transpose(x)
    
    return np.dot(Q,x)


def inverse_matrix(L,U,P,Q):
    E = np.eye(len(U))
    
    for i in range(len(U)):
        x = np.array(solve_LU(L,U,P,Q,E[:,i]))
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
    
    
    
    print('С невырожденной матрицей:')
    print()
    
    print('A: ')
    print(A)
      
    L, U, P, Q, permutation = LU_decomposition(A, True)
#     P1, L1, U1 = scipy.linalg.lu(A)
    
    print('P: ')
    print(P)
    print()
    print('Q:')
    print(Q)
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
    try:
        x = solve_LU(L,U,P,Q,B)
        print(x)
        print()
        print('Проверка Ax-b=0')
        print()
        print(np.dot(A,x)-B)
        print()
    except ValueError:
        print('Система несовместна')
        print()
    
    print('Обратная матрица:')
    print()
    print(inverse_matrix(L,U,P,Q))
    print()
    print(np.linalg.inv(A))
    print()
    print(f'Ранг матрицы через NumPy: {np.linalg.matrix_rank(A)}')
    print()
    print(f'Ранг матрицы через написанную функцию: {find_rank(A)}')
    print()
    
    print('С вырожденной матрицей:')
    print()
    
        #долго    
#     A = (-(2*n))*np.random.sample((n,n))+n
#     while np.linalg.det(A)!=0:
#         A = (-(2*n))*np.random.sample((n,n))+n
    A = np.array([
        [1,2,0,5,3],
        [2,6,0,7,8],
        [3,7,0,2,3],
        [1,6,0,6,8],
        [2,6,0,6,7]
    ])
    
    B = np.dot(A,B.copy())
    
    print(B)
    print()
    
    print('A: ')
    print(A)
      
    L, U, P, Q, permutation = LU_decomposition(A, True)
#     P1, L1, U1 = scipy.linalg.lu(A)
    
    print('P: ')
    print(P)
    print()
    print('Q:')
    print(Q)
    print()
    print('L: ')
    print(L)
    print()
    print('U: ')
    print(U)
    print()
    
#     print('P1: ')
#     print(P1)
#     print()
#     print('L1: ')
#     print(L1)
#     print()
#     print('U1: ')
#     print(U1)
#     print()
    
    print('PAQ и LU, соответственно:')
    print()
    print(np.dot(P,np.dot(A,Q)))
    print()
    print(np.dot(L,U))
    print()
    
    print('Определитель через встроенную функцию и через написанную выше, соответственно')
    print(f'NumPy determinant A = {np.linalg.det(A)}')
    print()
    print(f'Determinant A = {determinant(U, permutation)}')
    print()
    
    print('Решение СЛАУ через LU разложение: ')
    try:
        x = solve_LU(L,U,P,Q,B)
        print(x)
        print()
        print('Проверка Ax-b=0')
        print()
        Ax=np.dot(A,x)
        print(Ax)
        print()
    except ValueError:
        print('Система несовместна')
    
    print(f'Ранг матрицы через NumPy: {np.linalg.matrix_rank(A)}')
    print()
    print(f'Ранг матрицы через написанную функцию: {find_rank(A)}')
    print()

    
if __name__=='__main__':
    main()