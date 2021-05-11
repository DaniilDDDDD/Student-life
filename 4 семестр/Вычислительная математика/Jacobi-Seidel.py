import numpy as np
import random
import scipy
import scipy.linalg


def norm_m(A):
    m = []
    for i in range(len(A)):
        m.append(sum(abs(A[i])))
    return max(m)

def aposteriore(V1,V2,q):
    """используется норма бесконечности"""
    V1 = np.array(V1)
    V2 = np.array(V2)
    m = abs(V1-V2)
    return (q/(1-q))*max(m)

def apriore(V1,V2,q,k):
    """используется норма бесконечности"""
    V1 = np.array(V1)
    V2 = np.array(V2)
    m = abs(V1-V2)
    return ((q**k)/(1-q))*max(m)


# x - началльное приближение, e - точность, aposter - тип используемой оценки (True - апостериорная, иначе - априорная)
def solve_jacobi(A,b,e=10**(-9),aposter=True,need_k = False):
    R = np.triu(A,k=1)
    L = np.tril(A,k=-1)
    D_inv = np.linalg.inv(A - L - R)
    B = np.dot((-D_inv),(L+R))
    q = norm_m(B)
    if q >= 1:
        return -1
    C = np.dot(D_inv,b)
    previous_x = C
    x = np.dot(B,previous_x) + C
    k = 1
    
    if aposter:
        while abs(aposteriore(x,previous_x,q)) >= e:
            previous_x = x
            x = np.dot(B,previous_x) + C
            k += 1
            
    else:
        x_0 = previous_x
        x_1 = x
        while abs(apriore(x_1,x_0,q,k)) >= e:
            previous_x = x
            x = np.dot(B,previous_x) + C
            k += 1
    
    if need_k:
        return x,k
    return x

def solve_seidel(A,b,e=10**(-9),aposter=True,need_k = False, max_iter=5000):
    """Максимум 5000 итераций"""
    R = np.triu(A,k=1)
    L = np.tril(A,k=-1)
    D_inv = np.linalg.inv(A - L - R)
    B = np.dot((-D_inv),(L+R))
    q = norm_m(B)
    C = np.dot(D_inv,b)
    
    previous_x = C.copy()
    x = C.copy()
    for i in range(len(x)):
        x[i] = np.dot(B[i],x) + C[i]
    k = 1
    
    if aposter:
        while (abs(aposteriore(x,previous_x,q)) >= e) and (k<max_iter):
            previous_x = x.copy()
            for i in range(len(x)):
                x[i] = np.dot(B[i],x) + C[i]
            k += 1
            
    else:
        x_0 = previous_x
        x_1 = x
        while (abs(apriore(x_1,x_0,q,k))) >= e and (k<max_iter):
            previous_x = x.copy()
            for i in range(len(x)):
                x[i] = np.dot(B[i],x) + C[i]
            k += 1
            
    if need_k:
        return x,k
    return x
    

def main():
#     n = random.randint(0,100)
    n=4
    b = np.random.sample((n,1))
    X = np.random.sample((n,1))
    print('Для матрици с диагональным преобладанием:')
    print()
    
    '''Генерируем матрицу с диагональным преобладанием, со значениями не на диагонали от -n до n'''
    A = (-(2*n))*np.random.sample((n,n))+n
    for i in range(n):
        A[i][i] = A[i][i]*(n**2)
    R = np.triu(A,k=1)
    L = np.tril(A,k=-1)
    D_inv = np.linalg.inv(A - L - R)
    B = np.dot((-D_inv),(L+R))
    while (norm_m(B)>=1):
        A = (-(2*n))*np.random.sample((n,n))+n
        for i in range(n):
            A[i][i] = A[i][i]*(n**2)
        R = np.triu(A,k=1)
        L = np.tril(A,k=-1)
        D_inv = np.linalg.inv(A - L - R)
        B = np.dot((-D_inv),(L+R))

    print('b:')
    print(b)
    print()
    print('A:')
    print(A)
    print()
    print('scipy.linalg.solve:')
    print()
    print(scipy.linalg.solve(A,b))
    print()
    print('solve_jacobi с апостериорной оценкой:')
    print()
    x,k = solve_jacobi(A,b,need_k=True)
    print(x)
    print()
    print(f'Количество итераций: {k}')
    print()
    print('solve_jacobi с априорной оценкой:')
    print()
    x,k = solve_jacobi(A,b,aposter=False,need_k=True)
    print(x)
    print()
    print(f'Количество итераций: {k}')
    print()
    print('solve_seidel с апостериорной оценкой:')
    print()
    x,k = solve_seidel(A,b,need_k=True)
    print(x)
    print()
    print(f'Количество итераций: {k}')
    print()
    print('solve_seidel с априорной оценкой:')
    print()
    x,k = solve_seidel(A,b,aposter=False,need_k=True)
    print(x)
    print()
    print(f'Количество итераций: {k}')
    print()
    
    
    print('Для положительно определённой матрицы без диагонального преобладания и ||B||<1:')
    print()
    '''Генерация матрицы положительно определённой матрицы с ||B||<1'''
    A = np.random.sample((n,n))
    R = np.triu(A,k=1)
    L = np.tril(A,k=-1)
    D_inv = np.linalg.inv(A - L - R)
    B = np.dot((-D_inv),(L+R))
    while (norm_m(B)>=1):
        # генерируем положительные значения от 0 до n 
        A = n*np.random.sample((n,n))
        R = np.triu(A,k=1)
        L = np.tril(A,k=-1)
        D_inv = np.linalg.inv(A - L - R)
        B = np.dot((-D_inv),(L+R))
    print('A:')
    print(A)
    print()
    print('scipy.linalg.solve:')
    print()
    print(scipy.linalg.solve(A,b))
    print()
    print('solve_jacobi с апостериорной оценкой:')
    print()
    x,k = solve_jacobi(A,b,need_k=True)
    print(x)
    print()
    print(f'Количество итераций: {k}')
    print()
    print('solve_jacobi с априорной оценкой:')
    print()
    x,k = solve_jacobi(A,b,aposter=False,need_k=True)
    print(x)
    print()
    print(f'Количество итераций: {k}')
    print()
    print('solve_seidel с апостериорной оценкой:')
    print()
    x,k = solve_seidel(A,b,need_k=True)
    print(x)
    print()
    print(f'Количество итераций: {k}')
    print()
    print('solve_seidel с априорной оценкой:')
    print()
    x,k = solve_seidel(A,b,aposter=False,need_k=True)
    print(x)
    print()
    print(f'Количество итераций: {k}')
    print()
    
    print('Для положительно определённой, симметрической матрицы без диагонального преобладания и ||B||>=1:')
    print()
    '''Генерация матрицы положительно определённой матрицы с ||B||>=1'''
    A = np.random.sample((n,n))
    A = np.dot(A.transpose(),A)
    R = np.triu(A,k=1)
    L = np.tril(A,k=-1)
    D_inv = np.linalg.inv(A - L - R)
    B = np.dot((-D_inv),(L+R))
    while (norm_m(B)<1):
        # генерируем положительные значения от 0 до n 
        A = n*np.random.sample((n,n))
        A = np.dot(A.transpose(),A)
        R = np.triu(A,k=1)
        L = np.tril(A,k=-1)
        D_inv = np.linalg.inv(A - L - R)
        B = np.dot((-D_inv),(L+R))
    print('A:')
    print(A)
    print()
    print('scipy.linalg.solve:')
    print()
    print(scipy.linalg.solve(A,b))
    print()
    print('Метод Якоби не работает при ||B||>=1')
    print()
    print('solve_seidel с апостериорной оценкой:')
    print()
    x,k = solve_seidel(A,b,need_k=True)
    print(x)
    print()
    print(f'Количество итераций: {k}')
    print()
    print('solve_seidel с априорной оценкой:')
    print()
    x,k = solve_seidel(A,b,aposter=False,need_k=True)
    print(x)
    print()
    print(f'Количество итераций: {k}')
    print()


if __name__=='__main__':
    main()
