import numpy as np
import random
import scipy
import scipy.linalg



def Q_i(Q_min, i, j, k):
    """Construct the Q_t matrix by left-top padding the matrix Q                                                      
    with elements from the identity matrix."""
    if i < k or j < k:
        return float(i == j)
    else:
        return Q_min[i-k][j-k]

def householder(A):
    """Выполняет QR-разложение на основе теории Хаусхолдера для
     матрица A. Функция возвращает Q (ортогональная матрицу) и R
     (верхнетреугольная матрица) такие, что A = QR ."""
    n = len(A)

    R = A
    Q = np.zeros((n,n))

    # Процедура Хаусхолдера
    for k in range(n-1):  # Для матрицы 1 на 1 не рассматриваем
        
        I = np.eye(n)

        x = [row[k] for row in R[k:]]
        e = [row[k] for row in I[k:]]
        alpha = -(np.sign(x[0])) * np.linalg.norm(x)

        u = np.array(list(map(lambda p,q: p + alpha * q, x, e)))
        norm_u = np.linalg.norm(u)
        v = np.array(list(map(lambda p: p/norm_u, u)))

        Q_min = [ [float(i==j) - 2.0 * v[i] * v[j] for i in range(n-k)] for j in range(n-k) ]

        Q_t = [[ Q_i(Q_min,i,j,k) for i in range(n)] for j in range(n)]

        if k == 0:
            Q = Q_t
            R = np.dot(Q_t, A)
        else:
            Q = np.dot(Q_t, Q)
            R = np.dot(Q_t, R)

    return Q.transpose(), np.around(R,5)


def solve_QR(Q,R,b):
    """Rx=transpose(Q)b"""
    
    b = np.dot(Q.transpose(),b)
        
    x = [0]*len(R)
    x[len(R)-1] = (b[len(R)-1] / R[len(R)-1][len(R)-1])
    for i in reversed(range(len(R)-1)):
        cur = b[i]
        for j in range(i+1, len(R)):
            minus = 0
            minus += R[i][j]*x[j]
            cur -= minus
        x[i] = (cur/R[i][i])

    return np.array(x)
    


def main():
    #n = random.randint(0,100)
    n=5
    A = np.random.sample((n,n))
    B = np.random.sample((n,1))
    print('A:')
    print(A)
    print()
    print('B:')
    print(B)
    print()
    Q,R = householder(A)
    print('Написанная выше функция: точность до 5 знаков косле запятой')
    print('Q:')
    print(Q)
    print()
    print('R:')
    print(R)
    print()
    print('QR:')
    print(np.dot(Q,R))
    print()
    print('numpy.linalg.qr:')
    print()
    Q1,R1 = np.linalg.qr(A)
    print('Q:')
    print(Q1)
    print()
    print('R:')
    print(R1)
    print()
    print('solve_QR:')
    print()
    print(solve_QR(Q,R,B))
    print()
    print('scipy.linalg.solve:')
    print()
    print(scipy.linalg.solve(A,B))
    print()
    

if __name__=='__main__':
    main()