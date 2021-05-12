import numpy as np
import math
import time


def count_F(x):
    F = np.array([
        math.cos(x[1] * x[0]) - math.exp(-3 * x[2]) + x[3] * x[4] ** 2 - x[5] - math.sinh(2 * x[7]) * x[8] + 2 * x[
            9] + 2.000433974165385440,
        math.sin(x[1] * x[0]) + x[2] * x[8] * x[6] - math.exp(-x[9] + x[5]) + 3 * x[4] ** 2 - x[5] * (
                x[7] + 1) + 10.886272036407019994,
        x[0] - x[1] + x[2] - x[3] + x[4] - x[5] + x[6] - x[7] + x[8] - x[9] - 3.1361904761904761904,
        2 * math.cos(-x[8] + x[3]) + x[4] / (x[2] + x[0]) - math.sin(x[1] ** 2) + math.cos(x[6] * x[9]) ** 2 - x[
            7] - 0.1707472705022304757,
        math.sin(x[4]) + 2 * x[7] * (x[2] + x[0]) - math.exp(-x[6] * (-x[9] + x[5])) + 2 * math.cos(x[1]) - 1.0 / (
                -x[8] + x[3]) - 0.3685896273101277862,
        math.exp(x[0] - x[3] - x[8]) + x[4] ** 2 / x[7] + math.cos(3 * x[9] * x[1]) / 2 - x[5] * x[
            2] + 2.0491086016771875115,
        x[1] ** 3 * x[6] - math.sin(x[9] / x[4] + x[7]) + (x[0] - x[5]) * math.cos(x[3]) + x[2] - 0.7380430076202798014,
        x[4] * (x[0] - 2 * x[5]) ** 2 - 2 * math.sin(-x[8] + x[2]) + 0.15e1 * x[3] - math.exp(
            x[1] * x[6] + x[9]) + 3.5668321989693809040,
        7 / x[5] + math.exp(x[4] + x[3]) - 2 * x[1] * x[7] * x[9] * x[6] + 3 * x[8] - 3 * x[0] - 8.4394734508383257499,
        x[9] * x[0] + x[8] * x[1] - x[7] * x[2] + math.sin(x[3] + x[4] + x[5]) * x[6] - 0.78238095238095238096],
        dtype=float)

    return np.transpose(F)


def count_J(x):
    J = np.array([[-x[1] * math.sin(x[1] * x[0]), -x[0] * math.sin(x[1] * x[0]), 3 * math.exp(-3 * x[2]), x[4] ** 2,
                   2 * x[3] * x[4],
                   -1, 0, -2 * math.cosh(2 * x[7]) * x[8], -math.sinh(2 * x[7]), 2],
                  [x[1] * math.cos(x[1] * x[0]), x[0] * math.cos(x[1] * x[0]), x[8] * x[6], 0, 6 * x[4],
                   -math.exp(-x[9] + x[5]) - x[7] - 1, x[2] * x[8], -x[5], x[2] * x[6], math.exp(-x[9] + x[5])],
                  [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
                  [-x[4] / (x[2] + x[0]) ** 2, -2 * x[1] * math.cos(x[1] ** 2), -x[4] / (x[2] + x[0]) ** 2,
                   -2 * math.sin(-x[8] + x[3]),
                   1.0 / (x[2] + x[0]), 0, -2 * math.cos(x[6] * x[9]) * x[9] * math.sin(x[6] * x[9]), -1,
                   2 * math.sin(-x[8] + x[3]), -2 * math.cos(x[6] * x[9]) * x[6] * math.sin(x[6] * x[9])],
                  [2 * x[7], -2 * math.sin(x[1]), 2 * x[7], 1.0 / (-x[8] + x[3]) ** 2, math.cos(x[4]),
                   x[6] * math.exp(-x[6] * (-x[9] + x[5])), -(x[9] - x[5]) * math.exp(-x[6] * (-x[9] + x[5])),
                   2 * x[2] + 2 * x[0],
                   -1.0 / (-x[8] + x[3]) ** 2, -x[6] * math.exp(-x[6] * (-x[9] + x[5]))],
                  [math.exp(x[0] - x[3] - x[8]), -1.5 * x[9] * math.sin(3 * x[9] * x[1]), -x[5],
                   -math.exp(x[0] - x[3] - x[8]),
                   2 * x[4] / x[7], -x[2], 0, -x[4] ** 2 / x[7] ** 2, -math.exp(x[0] - x[3] - x[8]),
                   -1.5 * x[1] * math.sin(3 * x[9] * x[1])],
                  [math.cos(x[3]), 3 * x[1] ** 2 * x[6], 1, -(x[0] - x[5]) * math.sin(x[3]),
                   x[9] / x[4] ** 2 * math.cos(x[9] / x[4] + x[7]),
                   -math.cos(x[3]), x[1] ** 3, -math.cos(x[9] / x[4] + x[7]), 0,
                   -1.0 / x[4] * math.cos(x[9] / x[4] + x[7])],
                  [2 * x[4] * (x[0] - 2 * x[5]), -x[6] * math.exp(x[1] * x[6] + x[9]), -2 * math.cos(-x[8] + x[2]), 1.5,
                   (x[0] - 2 * x[5]) ** 2, -4 * x[4] * (x[0] - 2 * x[5]), -x[1] * math.exp(x[1] * x[6] + x[9]), 0,
                   2 * math.cos(-x[8] + x[2]),
                   -math.exp(x[1] * x[6] + x[9])],
                  [-3, -2 * x[7] * x[9] * x[6], 0, math.exp(x[4] + x[3]), math.exp(x[4] + x[3]),
                   -7.0 / x[5] ** 2, -2 * x[1] * x[7] * x[9], -2 * x[1] * x[9] * x[6], 3, -2 * x[1] * x[7] * x[6]],
                  [x[9], x[8], -x[7], math.cos(x[3] + x[4] + x[5]) * x[6], math.cos(x[3] + x[4] + x[5]) * x[6],
                   math.cos(x[3] + x[4] + x[5]) * x[6], math.sin(x[3] + x[4] + x[5]), -x[2], x[1], x[0]]], dtype=float)
    return J


'''x − sin x = 0.25'''


def F_scalar(x):
    return x - math.sin(x) - 0.25


'''d/dx(x − sin x = 0.25)'''


def J_scalar(x):
    return 1 - math.cos(x)


def LU_decomposition(A, find_permutation=False, count_operations=False):
    n = len(A)
    permutation = False
    C = np.array(A, dtype=float)
    P = np.eye(n)
    Q = np.eye(n)
    t = n - 1
    count = 6

    for i in range(n):
        pivotValue = 0
        pivot = -1
        count += 3

        while ((np.all(C[:, i] == 0)) and (i < t)):
            C[:, [i, t]] = C[:, [t, i]]
            Q[:, [i, t]] = Q[:, [t, i]]
            permutation = not permutation
            t -= 1
            count += 7

        for row in range(i, n):  # ищем максимальный элемент в столбце
            if (abs(C[row][i]) > pivotValue):
                pivotValue = abs(C[row][i])
                pivot = row
                count += 2
            count += 1

        if (pivotValue != 0):
            if pivot != i:
                P[[pivot, i]] = P[[i, pivot]]
                C[[pivot, i]] = C[[i, pivot]]
                permutation = not permutation
                count += 3
            count += 1
            for j in range(i + 1, n):
                C[j][i] /= C[i][i]
                count += 2
                for k in range(i + 1, n):
                    C[j][k] -= C[j][i] * C[i][k]
                    count += 3
        count += 1
        # теперь матрица C = L + U - E

    L = np.zeros((n, n))
    U = np.zeros((n, n))
    count += 2
    for i in range(n):
        for j in range(i):
            L[i][j] = C[i][j]
            count += 1
        L[i][i] = 1
        U[i][i] = C[i][i]
        count += 2
        for j in range(i + 1, n):
            U[i][j] = C[i][j]
            count += 1

    if find_permutation:
        if count_operations:
            return L, U, P, Q, permutation, count
        count += 1
        return L, U, P, Q, permutation
    if count_operations:
        return L, U, P, Q, count
    return L, U, P, Q


def solve_LU(L, U, P, Q, b, count_operations=False):
    """PLUx=Pb ; LU=PAQ => 1) Ly=Pb 2)Uz=y 3) x=Qz"""

    b = np.dot(P, b)

    y = [b[0]]
    count = 2
    for i in range(1, len(L)):
        cur = b[i]
        count += 1
        for j in range(i):
            minus = 0
            minus += L[i][j] * y[j]
            cur -= minus
            count += 5
        y.append(cur)
        count += 1

    rank = len(U)
    count += 1

    # проверка на совместность
    for i in range(len(U)):
        flag = True
        count += 1
        for j in U[i]:
            if abs(j) >= 1e-12:
                flag = False
                count += 1
                break
            count += 1
        if flag:
            rank -= 1
            count += 2
        if abs(y[i]) >= 10e-12 and flag:
            count += 2
            raise ValueError('Система несовместна')

    x = np.array([0] * len(U), dtype=float)
    x[rank - 1] = (y[rank - 1] / U[rank - 1][rank - 1])
    count += 3
    for i in reversed(range(rank - 1)):
        cur = y[i]
        count += 1
        for j in range(i + 1, rank):
            minus = 0
            minus += U[i][j] * x[j]
            cur -= minus
            count += 5
        x[i] = (cur / U[i][i])
        count += 2

    x = np.transpose(x)
    count += 2
    if count_operations:
        count += 1
        return np.dot(Q, x), count
    return np.dot(Q, x)


def inverse_matrix(L, U, P, Q, count_operations=False):
    E = np.eye(len(U))
    count = 1

    for i in range(len(U)):
        x, n = solve_LU(L, U, P, Q, E[:, i], count_operations=True)
        count += n
        E[:, i] = x.transpose()
        count += 3

    if count_operations:
        return E, count
    return E


def solve_newton(F, J, x, e=10 ** (-9), system=False, modified=0, period=1, count_iterations=False,
                 count_operations=False):
    """
    x_(k+1) = x^k - ([F'(x_k)]^-1)*F(x_k)
    
    Если modified==0, то сразу будет использован модифицированный метод
    """

    k = 1
    iterations = 1

    if not system:
        '''Для одного скалярного уравнения'''

        x_previous = x

        J_inv = 1 / J(x_previous)

        x = x_previous - F(x_previous) * J_inv

        k += 3

        while (abs(x - x_previous) >= e) and (iterations < modified):
            x_previous = x

            if iterations % period == 0:
                J_inv = 1 / J(x_previous)
                k += 2

            x = x_previous - (F(x_previous) * J_inv)
            k += 3
            iterations += 1

        while abs(x - x_previous) >= e:
            x_previous = x
            x = x_previous - (F(x_previous) * J_inv)
            k += 3
            iterations += 1

    else:
        '''Для системы скалярных уравнений'''

        x = np.array(x)
        x_previous = x.copy()

        k += 1

        L, U, P, Q, n = LU_decomposition(J(x_previous), count_operations=count_operations)

        temp, n = solve_LU(L, U, P, Q, -F(x_previous), count_operations=True)
        x = x_previous + temp
        k += n + 1

        while (np.linalg.norm((x - x_previous), ord=np.inf) >= e) and (iterations < modified):
            x_previous = x.copy()

            if iterations % period == 0:
                L, U, P, Q, n = LU_decomposition(J(x_previous), count_operations=True)
                k += n

            temp, n = solve_LU(L, U, P, Q, -F(x_previous), count_operations=True)
            x = x_previous + temp
            iterations += 1
            k += 1 + n

        while np.linalg.norm((x - x_previous), ord=np.inf) >= e:
            x_previous = x.copy()
            temp, n = solve_LU(L, U, P, Q, -F(x_previous), count_operations=True)
            x = x_previous + temp
            iterations += 1
            k += 1 + n

    if count_iterations:
        if count_operations:
            return x, iterations, k
        return x, iterations
    if count_operations:
        return x, k
    return x


def main():
    print('Номер 1:')
    print()

    x_0 = 1.0

    print('Немодифицированный метод Ньютона:')
    print()

    timestamp = time.time()

    try:
        x, iterations, operations = solve_newton(F_scalar, J_scalar, x_0, e=0.0001, system=False, modified=500,
                                                 period=1,
                                                 count_iterations=True, count_operations=True)

        print('x:')
        print(x)
        print()

        print(f'Время работы: {time.time() - timestamp}')
        print()

        print(f'Количество итераций: {iterations}')
        print()

        print(f'Количество операций: {operations}')
        print()

        print('Проверка F(x)==0:')
        print(F_scalar(x))
        print()

    except OverflowError:
        print('С данными значениям метод не сходится!')
        print(f'Время работы: {time.time() - timestamp}')
        print()

    print('Модифицированный метод Ньютона:')
    print()

    timestamp = time.time()

    try:
        x, iterations, operations = solve_newton(F_scalar, J_scalar, x_0, e=0.0001, system=False, modified=0,
                                                 count_iterations=True, count_operations=True)

        print('x:')
        print(x)
        print()

        print(f'Время работы: {time.time() - timestamp}')
        print()

        print(f'Количество итераций: {iterations}')
        print()

        print(f'Количество операций: {operations}')
        print()

        print('Проверка F(x)==0:')
        print(F_scalar(x))
        print()
        print()
        print()

    except OverflowError:
        print('С данными значениям метод не сходится!')
        print(f'Время работы: {time.time() - timestamp}')
        print()
        print()
        print()

    print('Номер 2:')
    print()

    x_0 = np.transpose(np.array([0.5, 0.5, 1.5, -1.0, -0.5, 1.5, 0.5, -0.5, 1.5, -1.5]))

    print('Немодифицированный метод Ньютона:')
    print()

    timestamp = time.time()
    try:
        # В данном случае при 2х итерациях полным методом - наименьшее время рассчёта
        x, iterations, operations = solve_newton(count_F, count_J, x_0, system=True, modified=500, period=10,
                                                 count_iterations=True, count_operations=True)

        print('x:')
        print(x)
        print()

        print(f'Проверка F(x)==0:')
        print(count_F(x))
        print()

        print(f'Время работы: {time.time() - timestamp}')
        print()

        print(f'Количество итераций: {iterations}')
        print()

        print(f'Количество операций: {operations}')
        print()

    except OverflowError:
        print('С данными значениям метод не сходится!')
        print(f'Время работы: {time.time() - timestamp}')
        print()

    print('Модифицированный метод Ньютона:')
    print()

    timestamp = time.time()
    x, iterations, operations = solve_newton(count_F, count_J, x_0, system=True, modified=0, count_iterations=True,
                                             count_operations=True)

    print('x:')
    print(x)
    print()

    print(f'Проверка F(x)==0:')
    print(count_F(x))
    print()

    print(f'Время работы: {time.time() - timestamp}')
    print()

    print(f'Количество итераций: {iterations}')
    print()

    print(f'Количество операций: {operations}')
    print()
    print()

    x_0 = np.transpose(np.array([0.5, 0.5, 1.5, -1.0, -0.2, 1.5, 0.5, -0.5, 1.5, -1.5]))

    print('Немодифицированный метод Ньютона: (x_0[5] = -0.2)')
    print()

    print('modified = 4 (<7):')
    print()
    print()

    print('period = 4 (<7):')
    print()
    try:
        timestamp = time.time()

        x, iterations, operations = solve_newton(count_F, count_J, x_0, system=True, modified=100, period=4,
                                                 count_iterations=True, count_operations=True)

        print('x:')
        print(x)
        print()

        print(f'Проверка F(x)==0:')
        print(count_F(x))
        print()

        print(f'Время работы: {time.time() - timestamp}')
        print()

        print(f'Количество итераций: {iterations}')
        print()

        print(f'Количество операций: {operations}')
        print()
        print()
    except OverflowError:
        print('С данными значениям метод не сходится!')
        print(f'Время работы: {time.time() - timestamp}')
        print()
        print()

    print('period = 7:')
    print()

    timestamp = time.time()

    try:
        x, iterations, operations = solve_newton(count_F, count_J, x_0, system=True, modified=4, period=7,
                                                 count_iterations=True, count_operations=True)
        print('x:')
        print(x)
        print()

        print(f'Проверка F(x)==0:')
        print(count_F(x))
        print()

        print(f'Время работы: {time.time() - timestamp}')
        print()

        print(f'Количество итераций: {iterations}')
        print()

        print(f'Количество операций: {operations}')
        print()
        print()

    except OverflowError:
        print('С данными значениям метод не сходится!')
        print(f'Время работы: {time.time() - timestamp}')
        print()
        print()

    print('period = 10 (>7):')
    print()

    try:
        timestamp = time.time()
        x, iterations, operations = solve_newton(count_F, count_J, x_0, system=True, modified=4, period=10,
                                                 count_iterations=True, count_operations=True)

        print('x:')
        print(x)
        print()

        print(f'Проверка F(x)==0:')
        print(count_F(x))
        print()

        print(f'Время работы: {time.time() - timestamp}')
        print()

        print(f'Количество итераций: {iterations}')
        print()

        print(f'Количество операций: {operations}')
        print()
        print()

        print('modified = 7:')
        print()
        print()

    except OverflowError:
        print('С данными значениям метод не сходится!')
        print(f'Время работы: {time.time() - timestamp}')
        print()
        print()

    print('period = 4 (<7):')
    print()
    timestamp = time.time()

    try:
        x, iterations, operations = solve_newton(count_F, count_J, x_0, system=True, modified=7, period=4,
                                                 count_iterations=True, count_operations=True)

        print('x:')
        print(x)
        print()

        print(f'Проверка F(x)==0:')
        print(count_F(x))
        print()

        print(f'Время работы: {time.time() - timestamp}')
        print()

        print(f'Количество итераций: {iterations}')
        print()

        print(f'Количество операций: {operations}')
        print()
        print()


    except OverflowError:
        print('С данными значениям метод не сходится!')
        print(f'Время работы: {time.time() - timestamp}')
        print()
        print()

    print('period = 7:')
    print()
    timestamp = time.time()

    try:
        x, iterations, operations = solve_newton(count_F, count_J, x_0, system=True, modified=7, period=7,
                                                 count_iterations=True, count_operations=True)

        print('x:')
        print(x)
        print()

        print(f'Проверка F(x)==0:')
        print(count_F(x))
        print()

        print(f'Время работы: {time.time() - timestamp}')
        print()

        print(f'Количество итераций: {iterations}')
        print()

        print(f'Количество операций: {operations}')
        print()
        print()

    except OverflowError:
        print('С данными значениям метод не сходится!')
        print(f'Время работы: {time.time() - timestamp}')
        print()
        print()

    print('period = 10 (>7):')
    print()

    timestamp = time.time()

    try:
        x, iterations, operations = solve_newton(count_F, count_J, x_0, system=True, modified=7, period=10,
                                                 count_iterations=True, count_operations=True)

        print('x:')
        print(x)
        print()

        print(f'Проверка F(x)==0:')
        print(count_F(x))
        print()

        print(f'Время работы: {time.time() - timestamp}')
        print()

        print(f'Количество итераций: {iterations}')
        print()

        print(f'Количество операций: {operations}')
        print()
        print()

    except OverflowError:
        print('С данными значениям метод не сходится!')
        print(f'Время работы: {time.time() - timestamp}')
        print()
        print()

    print('modified = 15 (>7):')
    print()
    print()

    print('period = 4 (<7):')
    print()
    timestamp = time.time()

    try:
        x, iterations, operations = solve_newton(count_F, count_J, x_0, system=True, modified=15, period=4,
                                                 count_iterations=True, count_operations=True)

        print('x:')
        print(x)
        print()

        print(f'Проверка F(x)==0:')
        print(count_F(x))
        print()

        print(f'Время работы: {time.time() - timestamp}')
        print()

        print(f'Количество итераций: {iterations}')
        print()

        print(f'Количество операций: {operations}')
        print()
        print()

    except OverflowError:
        print('С данными значениям метод не сходится!')
        print(f'Время работы: {time.time() - timestamp}')
        print()
        print()

    print('period = 7:')
    print()

    timestamp = time.time()

    try:

        x, iterations, operations = solve_newton(count_F, count_J, x_0, system=True, modified=15, period=7,
                                                 count_iterations=True, count_operations=True)

        print('x:')
        print(x)
        print()

        print(f'Проверка F(x)==0:')
        print(count_F(x))
        print()

        print(f'Время работы: {time.time() - timestamp}')
        print()

        print(f'Количество итераций: {iterations}')
        print()

        print(f'Количество операций: {operations}')
        print()
        print()

    except OverflowError:
        print('С данными значениям метод не сходится!')
        print(f'Время работы: {time.time() - timestamp}')
        print()
        print()

    print('period = 10 (>7):')
    print()

    timestamp = time.time()

    try:
        x, iterations, operations = solve_newton(count_F, count_J, x_0, system=True, modified=15, period=10,
                                                 count_iterations=True, count_operations=True)

        print('x:')
        print(x)
        print()

        print(f'Проверка F(x)==0:')
        print(count_F(x))
        print()

        print(f'Время работы: {time.time() - timestamp}')
        print()

        print(f'Количество итераций: {iterations}')
        print()

        print(f'Количество операций: {operations}')
        print()
        print()
    except OverflowError:
        print('С данными значениям метод не сходится!')
        print(f'Время работы: {time.time() - timestamp}')
        print()
        print()


if __name__ == '__main__':
    main()
