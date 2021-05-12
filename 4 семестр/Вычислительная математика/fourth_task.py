import numpy as np
from numpy import linalg
from numpy.random import rand
import math

a0 = 1.5
b0 = 2.3
alpha = 0.2
beta = 0
eps = 1e-6
real = 32.21951452884267


def f(x):
    return 2 * math.cos(3.5 * x) * math.exp(5.0 / 3.0 * x) + 3 * math.sin(1.5 * x) * math.exp(-4 * x) + 3


def m0(a, b):
    return (5 * ((2 * b - 3) ** (4 / 5) - (2 * a - 3) ** (4 / 5))) / 2 ** (14 / 5)


def m1(a, b):
    return ((2 * b - 3) ** (4 / 5) * (40 * b + 75) + (-40 * a - 75) * (2 * a - 3) ** (4 / 5)) / (9 * 2 ** (19 / 5))


def m2(a, b):
    return ((2 * b - 3) ** (4 / 5) * (120 * b ** 2 + 200 * b + 375) + (2 * a - 3) ** (4 / 5) * (
                -120 * a ** 2 - 200 * a - 375)) / (21 * 2 ** (24 / 5))


def m3(a, b):
    return ((2 * b - 3) ** (4 / 5) * (1120 * b ** 3 + 1800 * b ** 2 + 3000 * b + 5625) + (2 * a - 3) ** (4 / 5) * (
                -1120 * a ** 3 - 1800 * a ** 2 - 3000 * a - 5625)) / (133 * 2 ** (29 / 5))


def m4(a, b):
    return ((2 * b - 3) ** (4 / 5) * (10640 * b ** 4 + 16800 * b ** 3 + 27000 * b ** 2 + 45000 * b + 84375) + (
                2 * a - 3) ** (4 / 5) * (-10640 * a ** 4 - 16800 * a ** 3 - 27000 * a ** 2 - 45000 * a - 84375)) / (
                       399 * 2 ** (39 / 5))


def m5(a, b):
    return ((2 * b - 3) ** (4 / 5) * (
                170240 * b ** 5 + 266000 * b ** 4 + 420000 * b ** 3 + 675000 * b ** 2 + 1125000 * b + 2109375) + (
                        2 * a - 3) ** (4 / 5) * (
                        -170240 * a ** 5 - 266000 * a ** 4 - 420000 * a ** 3 - 675000 * a ** 2 - 1125000 * a - 2109375)) / (
                       3857 * 2 ** (44 / 5))


# использую функции для решения СЛАУ из второго задания

# чтобы избежать сильного влияния вычислительной погрешности на решение, применяем метод Гаусса с выбором главного элемента
# среди элементов матрицы  выбераем наибольший по модулю(главный)элемент
def find_P(a, row, column):
    shape = a.shape[0]
    maximum = 0
    # найдем максимальный элемент в столбце и его позицию
    for item in range(row, shape):
        if abs(a[item][column]) >= abs(maximum):
            maximum = a[item][column]
            max_row = item

    # если максимум равен нулю, то вернем False
    if abs(maximum) < 1e-14:
        return False

    # если не равен нулю, то выведем матрицу перестановок
    result_matrix = np.eye(shape)  # создаем матрицу нулей

    result_matrix[row][row] = 0
    result_matrix[max_row][max_row] = 0
    result_matrix[row][max_row] = 1
    result_matrix[max_row][row] = 1
    return result_matrix


# Находим матрицу M из формулы
def get_M(a, row, column, shape):
    result_matrix = np.eye(shape)
    for item in range(row + 1, shape):
        result_matrix[item][row] = -a[item][column] / a[row][column]
    return result_matrix


# LUP разложение
def LUP_decomposition(a):
    A = a.copy()
    m, n = A.shape
    shape = m

    M_total = np.eye(shape)

    row = 0
    P_total = np.eye(shape)
    for column in range(shape):
        P = find_P(A, row, column)

        # Если максимальный элемент в строке не 0, то переставляем ее на верх и вычитаем из всех строк ниже
        if np.any(P):
            A = np.dot(P, A)
            P_total = np.dot(P_total, P)
            M = get_M(A, row, column, shape)  # Подбираем M
            M_total = np.dot(M, np.dot(P,
                                       M_total))  # Собираем результирующую M           Умножаем слева промежуточную M на произведение P*M_total
            A = np.dot(M, A)  # Получаем Промежуточный результат                            Наша будущая матрица U
            row += 1

    p = linalg.inv(P_total)
    l = np.dot(p, linalg.inv(M_total))
    u = A
    return l, u, p


# определитель матрицы
def det(a):
    shape = a.shape[0]
    L, U, P = LUP_decomposition(a)

    n_transposition = 1
    for i in range(shape):
        if P[i][i] != 1:
            n_transposition += 1

    result = 1
    for i in range(shape):
        result *= U[i][i]
    return result * (-1) ** n_transposition


def solveLU(A, b):
    L, U, P = LUP_decomposition(A)
    shape = L.shape[0]
    B = np.array(b)
    B = np.dot(P, B)

    y = [B[0]]  # находим вектор y прямым ходом метода Гауса
    for i in range(1, shape):
        y.append(B[i])
        for item in range(i):
            y[i] -= y[item] * L[i][item]

    if abs(det(A)) > 1e-14:  # проверяем вырожденность матрицы

        x = np.zeros(shape)  # находим вектор x обратным ходом метода Гауса
        x[shape - 1] = y[shape - 1] / U[shape - 1][shape - 1]

        for i in range(shape - 2, -1, -1):
            x[i] = y[i]

            for item in range(i + 1, shape):
                x[i] -= U[i][item] * x[item]

            x[i] /= U[i][i]

    else:
        # поиск ранга матрицы
        rank = 0
        for row in range(shape):
            flag = False
            for column in range(shape):
                if abs(U[row][column]) > 1e-14:
                    flag = True
            if flag:
                rank += 1

        for i in range(shape - 1, rank, -1):
            if abs(y[i]) > 1e-14:
                return False

        x = np.zeros(shape)  # находим вектор x обратным ходом метода Гауса
        for row in range(rank - 1, -1, -1):
            col = 0
            flag = True
            while flag and col < shape - 1:
                if abs(U[row][col]) > 1e-14:
                    flag = False
                else:
                    col += 1

            x[col] = y[row]
            for item in range(col + 1, shape):
                x[col] -= U[row][item] * x[item]
            x[col] /= U[row][col]
    return x


# Ньютон-Котс
def NewtonCat(a, b, h):
    step = (b - a) / h
    integral = 0.0
    for i in range(h):
        b = a + step

        x = np.array([a, a + (b - a) / 2, b], float)
        m = np.array([m0(a, b), m1(a, b), m2(a, b)], float)

        matrix = np.array([[1, 1, 1], [x[0], x[1], x[2]], [x[0] ** 2, x[1] ** 2, x[2] ** 2]], float)

        A = solveLU(matrix, m)
        for i in range(3):
            integral += A[i] * f(x[i])

        a += step
    return integral


def Kardano(a, x):
    p = a[1] - a[2] * a[2] / 3.0
    q = a[0] + 2.0 * a[2] * a[2] * a[2] / 27.0 - a[2] * a[1] / 3.0
    D = q * q / 4.0 + p * p * p / 27.0
    if D < 0:
        fi = 0
        if q < 0:
            fi = math.atan(2.0 * (-D) ** 0.5 / (-q))
        if q > 0:
            fi = math.atan(2.0 * (-D) ** 0.5 / (-q) + math.pi)
        if q == 0:
            fi = math.pi / 2.0
        x[0] = 2.0 * (-p / 3.0) ** 0.5 * math.cos(fi / 3.0) - a[2] / 3.0
        x[1] = 2.0 * (-p / 3.0) ** 0.5 * math.cos(fi / 3.0 + 2.0 * math.pi / 3.0) - a[2] / 3.0
        x[2] = 2.0 * (-p / 3.0) ** 0.5 * math.cos(fi / 3.0 + 4.0 * math.pi / 3.0) - a[2] / 3.0
    elif D > 0:
        x[1] = 0
        if (-q) / 2.0 + pow(D, 1.0 / 2.0) < 0:
            x[1] += -pow(q / 2.0 - pow(D, 1.0 / 2.0), 1.0 / 3.0)
        else:
            x[1] += pow((-q) / 2.0 + pow(D, 1.0 / 2.0), 1.0 / 3.0)
        if -q / 2.0 - pow(D, 1.0 / 2.0) < 0:
            x[1] += -pow(q / 2.0 + pow(D, 1.0 / 2.0), 1.0 / 3.0) - a[2] / 3.0
        else:
            x[1] += pow(-q / 2.0 - pow(D, 1.0 / 2.0), 1.0 / 3.0) - a[2] / 3.0
    else:
        x[0] = 2 * pow(-q / 2.0, 1.0 / 3.0) - a[2] / 3.0
        x[1] = -pow(-q / 2.0, 1.0 / 3.0) - a[2] / 3.0
    return x


def Gauss(a, b, h):
    bl = b
    b = a + h
    integral = 0.0
    k = math.ceil((bl - b) / h) + 2
    for i in range(1, k, 1):

        m = np.array([m0(a, b), m1(a, b), m2(a, b), m3(a, b), m4(a, b), m5(a, b)], float)
        matrix = np.array([[m[0], m[1], m[2]], [m[1], m[2], m[3]], [m[2], m[3], m[4]]], dtype=float)
        y = np.array([-m[3], -m[4], -m[5]], float)
        A = solveLU(matrix, y)

        x = np.array([a, 0, b], float)
        x = Kardano(A, x)  # находим узлы
        x = sorted(x)

        A = np.array([[1, 1, 1], [x[0], x[1], x[2]], [x[0] * x[0], x[1] * x[1], x[2] * x[2]]], float)
        res = solveLU(A, m[:3])

        for j in range(3):
            integral += res[j] * f(x[j])
        a = b
        b += h

    return integral


# Вывод:
print('\n')
print('Формула Ньютона-Котса: ')
cat_mistake = 2.768908  # посчитала в вольфраме
cat_result = NewtonCat(a0, b0, 1)
print('Результат: ', cat_result)
print('Методическая погрешность: ', cat_mistake)
print('Точная погрешность: ', abs(real - cat_result))
print('\n')

e = 100000000
h = 1
L = 2
d = 4
print('Скорость сходимости составной ИКФ: ')
while e > eps:
    res = np.zeros(3)
    for i in range(3):
        res[i] = NewtonCat(a0, b0, h)
        h = h * L
    # оцениваем скорость сходимости
    speed = - math.log((res[2] - res[1]) / (res[1] - res[0])) / math.log(L)
    print(speed)

    e = abs((res[1] - res[0]) / (math.pow(L, d) - 1))
    ikf_result = res[0] + (res[1] - res[0]) / (1 - (1 / math.pow(L, d)))

print('Составная ИКФ: ')
print('Результат: ', ikf_result)
print('Точная погрешность: ', abs(real - ikf_result))
print('Методическая погрешность: ', e)
print('\n')

e = 10
h_opt = math.ceil((b0 - a0) / (((b0 - a0) / 2) * L * math.pow((0.00001 / e), 1.0 / d)))
cat_result_opt = NewtonCat(a0, b0, h_opt)
e = abs((cat_result_opt - ikf_result) / (math.pow(L, d) - 1))

print('Составная ИКФ c оптимальным шагом: ')
print('Результат: ', cat_result_opt)
print('Точная погрешность: ', abs(real - cat_result_opt))
print('Методическая погрешность ', e)
print('\n')

print('Формула Гаусса: ')
gauss_mistake = 33.367  # посчитала в вольфраме
gauss_result = Gauss(a0, b0, 1)
print('Результат: ', gauss_result)
print('Методическая погрешность: ', gauss_mistake)
print('Точная погрешность: ', abs(real - gauss_result))
print('\n')

h = (b0 - a0) / 10
e = 10
res0 = Gauss(a0, b0, h)
print('Скорость сходимости составной ИКФ Гаусса: ')
while e > eps:
    res = np.array([float(0) for i in range(3)])
    h *= L
    res[0] = Gauss(a0, b0, h)
    h /= L
    res[1] = Gauss(a0, b0, h)
    h /= L
    res[2] = Gauss(a0, b0, h)
    speed = -math.log(abs((res[2] - res[1]) / (res[1] - res[0]))) / math.log(L)
    print(speed)
    e = abs((res[1] - res[0]) / (math.pow(L, d) - 1))
    ikf_result += (res[1] - res[0]) / (1 - math.pow(L, -d))

print('Составная ИКФ Гаусса: ')
print('Результат: ', ikf_result)
print('Точная погрешность: ', abs(real - ikf_result))
print('Методическая погрешность: ', e)

print('\n')

# вычисляем оптимальный шаг
h_opt = math.ceil((b0 - a0) / (h * L * pow((0.00001 / e), 1.0 / d)))
h_opt = (b0 - a0) / (h_opt)
# print(h_opt)


gauss_result = Gauss(a0, b0, h_opt)
e = abs((gauss_result - res0) / (pow(L, d) - 1))

print('Составная ИКФ Гаусса с оптимальным шагом: ')
print('Результат: ', gauss_result)
print('Точная погрешность: ', abs(real - gauss_result))
print('Методическая погрешность: ', e)
print('\n')
