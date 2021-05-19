import math
import numpy as np


def newton_cots(f, a, b, moments, optimal=False, step_number=2, r=3,
                error=10 ** (-6)):  # метод Ньютона-Котса
    k = -1
    L = 2
    S = np.array([], dtype=float)
    H_begin = np.array([], dtype=float)

    # первые 3 итерации поиска интеграла (необходимы для использования метода Ричардсона)
    for j in range(r):
        h = round((b - a) / (step_number * (L ** j)), 10)
        H_begin = np.append(H_begin, h)
        x = np.arange(a, b, h)
        x = np.append(x, b)
        s = 0
        for i in range(len(x) - 1):
            current_interval = np.array([x[i], (x[i + 1] + x[i]) / 2, x[i + 1]], dtype=float)
            moments_values = [m(current_interval[0], current_interval[2]) for m in moments]
            xes = np.array([current_interval ** i for i in range(len(current_interval))], dtype=float)
            A = np.linalg.solve(xes, moments_values)
            s += sum([A[m] * f(current_interval[m]) for m in range(len(current_interval))])
        S = np.append(S, s)

    # Метод Эйткена

    if S[2] != S[1] and S[0] != S[1]:
        m = -((math.log(abs((S[2] - S[1]) / (S[1] - S[0])))) / math.log(L))
    else:
        return S[-1], 0

    # По правилу Рунге
    R = np.array([], dtype=float)
    R = np.append(R, (S[-1] - S[-2]) / (1 - L ** (-m)))
    R = np.append(R, (S[-1] - S[-2]) / ((L ** m) - 1))

    if not optimal:  # без поиска оптимального шага h
        while abs(R[-1]) >= error:
            h = round(H_begin[-1] / L, 10)
            H_begin = np.append(H_begin, h)
            x = np.arange(a, b, h)
            x = np.append(x, b)
            s = 0
            for i in range(len(x) - 1):
                current_interval = np.array([x[i], (x[i + 1] + x[i]) / 2, x[i + 1]], dtype=float)
                moments_values = [m(current_interval[0], current_interval[2]) for m in moments]
                xes = np.array([current_interval ** i for i in range(len(current_interval))], dtype=float)
                A = np.linalg.solve(xes, moments_values)
                s += sum([A[m] * f(current_interval[m]) for m in range(len(current_interval))])
            S = np.append(S, s)

            if S[-1] != S[-2]:
                m = -((math.log(abs((S[-1] - S[-2]) / (S[-2] - S[-3])))) / math.log(L))
                # print(m)
                if m < len(moments):
                    continue
            else:
                break

            R = np.append(R, (S[-1] - S[-2]) / (1 - L ** (-m)))
            R = np.append(R, (S[-1] - S[-2]) / ((L ** m) - 1))

    else:  # с поиском оптимального шага h
        while abs(R[-1]) >= error:
            h = round(H_begin[-1] * ((error / abs(R[-1])) ** (1 / m)), 10)

            H_begin = np.append(H_begin, h * (L ** 2))
            H_begin = np.append(H_begin, h * L)
            H_begin = np.append(H_begin, h)

            for j in reversed(range(3)):
                s = 0
                x = np.arange(a, b, H_begin[-j - 1])
                x = np.append(x, b)

                for i in range(len(x) - 1):
                    current_interval = np.array([x[i], (x[i + 1] + x[i]) / 2, x[i + 1]], dtype=float)
                    moments_values = [m(current_interval[0], current_interval[2]) for m in moments]
                    xes = np.array([current_interval ** i for i in range(len(current_interval))], dtype=float)
                    A = np.linalg.solve(xes, moments_values)
                    s += sum([A[m] * f(current_interval[m]) for m in range(len(current_interval))])
                S = np.append(S, s)

            if S[-1] != S[-2]:
                m_1 = -((math.log(abs((S[-1] - S[-2]) / (S[-2] - S[-3])))) / math.log(L))
                if m_1 < len(moments) - 0.5:
                    continue
                else:
                    m = m_1
            else:
                break

            R = np.append(R, (S[-1] - S[-2]) / (1 - L ** (-m)))
            R = np.append(R, (S[-1] - S[-2]) / ((L ** m) - 1))

    # print((b - a) / H_begin[-1])
    return S[-1], R[-1]


def gauss(f, a, b, moments, optimal=False, step_number=1, r=3, error=10 ** (-9)):
    S = np.array([], dtype=float)
    H_begin = np.array([], dtype=float)
    L = 2
    k = -1
    n = int(len(moments) / 2)

    for k in range(r):
        h = round((b - a) / (step_number * (L ** k)), 10)
        H_begin = np.append(H_begin, h)
        x = np.arange(a, b, h)
        x = np.append(x, b)
        s = 0
        for i in range(len(x) - 1):
            moments_values = [m(x[i], x[i + 1]) for m in moments]
            matrix = [[moments_values[(j + s)] for j in range(n)] for s in range(n)]

            right_part = [-moments_values[n + s] for s in range(n)]
            w_coefficients = np.linalg.solve(matrix, right_part)

            w_coefficients = [1] + [w_coefficients[i] for i in reversed(range(len(w_coefficients)))]
            roots = np.roots(w_coefficients)

            matrix = [[roots[j] ** s for j in range(n)] for s in range(n)]
            A = np.linalg.solve(matrix, moments_values[:n])

            s += sum([A[j] * f(roots[j]) for j in range(n)])

        S = np.append(S, s)

    if S[-1] != S[-2]:
        m = -((math.log(abs((S[-1] - S[-2]) / (S[-2] - S[-3])))) / math.log(L))
    else:
        return S[-1], 0

    # По правилу Рунге
    R = np.array([], dtype=float)
    R = np.append(R, (S[-1] - S[-2]) / (1 - L ** (-m)))
    R = np.append(R, (S[-1] - S[-2]) / ((L ** m) - 1))

    if not optimal:  # без поиска оптимального шага h
        while abs(R[-1]) >= error:
            h = round(H_begin[-1] / L, 10)
            H_begin = np.append(H_begin, h)
            x = np.arange(a, b, h)
            x = np.append(x, b)
            s = 0
            for i in range(len(x) - 1):
                moments_values = [m(x[i], x[i + 1]) for m in moments]
                matrix = [[moments_values[j + s] for j in range(n)] for s in range(n)]
                right_part = [-moments_values[n + s] for s in range(n)]
                w_coefficients = np.linalg.solve(matrix, right_part)

                w_coefficients = [1] + [w_coefficients[i] for i in reversed(range(len(w_coefficients)))]
                roots = np.roots(w_coefficients)

                matrix = [[roots[j] ** s for j in range(n)] for s in range(n)]
                A = np.linalg.solve(matrix, moments_values[:n])

                s += sum([A[j] * f(roots[j]) for j in range(n)])

            S = np.append(S, s)

            if S[-1] != S[-2]:
                m = -((math.log(abs((S[-1] - S[-2]) / (S[-2] - S[-3])))) / math.log(L))
                # print(m)
                if m < len(moments):
                    continue
            else:
                return S[-1], R[-1]

            R = np.array([], dtype=float)
            R = np.append(R, (S[-1] - S[-2]) / (1 - L ** (-m)))
            R = np.append(R, (S[-1] - S[-2]) / ((L ** m) - 1))

    else:  # с поиском оптимального шага h
        while abs(R[-1]) >= error:
            h = round(H_begin[-1] * ((error / abs(R[-1])) ** (1 / m)), 10)
            # print((b-a)/h, ' h')

            H_begin = np.append(H_begin, h * (L ** 2))
            H_begin = np.append(H_begin, h * L)
            H_begin = np.append(H_begin, h)

            for j in reversed(range(3)):

                x = np.arange(a, b, H_begin[-j - 1])
                x = np.append(x, b)
                s = 0
                for i in range(len(x) - 1):
                    moments_values = [m(x[i], x[i + 1]) for m in moments]
                    matrix = [[moments_values[j + s] for j in range(n)] for s in range(n)]
                    right_part = [-moments_values[n + s] for s in range(n)]
                    w_coefficients = np.linalg.solve(matrix, right_part)

                    w_coefficients = [1] + [w_coefficients[i] for i in reversed(range(len(w_coefficients)))]
                    roots = np.roots(w_coefficients)

                    matrix = [[roots[j] ** s for j in range(n)] for s in range(n)]
                    A = np.linalg.solve(matrix, moments_values[:n])

                    s += sum([A[j] * f(roots[j]) for j in range(n)])
                S = np.append(S, s)

            if S[-1] != S[-2]:
                m_1 = -((math.log(abs((S[-1] - S[-2]) / (S[-2] - S[-3])))) / math.log(L))
                if m_1 < len(moments) - 1:
                    continue
                else:
                    m = m_1
                    # print(m)
            else:
                break

            R = np.append(R, (S[-1] - S[-2]) / (1 - L ** (-m)))
            R = np.append(R, (S[-1] - S[-2]) / ((L ** m) - 1))
    # print((b - a) / H_begin[-1])

    return S[-1], R[-1]


def F(x):
    return 4 * math.cos(2.5 * x) * math.exp((4 * x) / 7) + 2.5 * math.sin(5.5 * x) * math.exp(-(3 * x) / 5) + 4.3 * x


def m0(a, b):
    return (-7 / 3) * ((((29 / 10) - b) ** (3 / 7)) - (((29 / 10) - a) ** (3 / 7)))


def m1(a, b):
    return (-7 / 300) * ((30 * b + 203) * (((29 / 10) - b) ** (3 / 7)) - (30 * a + 203) * (((29 / 10) - a) ** (3 / 7)))


def m2(a, b):
    return (-7 / 25500) * (
            (1500 * b ** 2 + 6090 * b + 41209) * (((29 / 10) - b) ** (3 / 7)) - (1500 * a ** 2 + 6090 * a + 41209) * (
            ((29 / 10) - a) ** (3 / 7)))


def m3(a, b):
    return (-7 / 2040000) * (
            (85000 * b ** 3 + 304500 * b ** 2 + 1236270 * b + 8365427) * (((29 / 10) - b) ** (3 / 7)) - (
            85000 * a ** 3 + 304500 * a ** 2 + 1236270 * a + 8365427) * (((29 / 10) - a) ** (3 / 7)))


def m4(a, b):
    return (-7 / 158100000) * (
            (5100000 * b ** 4 + 17255000 * b ** 3 + 61813500 * b ** 2 + 250962810 * b + 1698181681) * (
            ((29 / 10) - b) ** (3 / 7)) - (
                    5100000 * a ** 4 + 17255000 * a ** 3 + 61813500 * a ** 2 + 250962810 * a + 1698181681) * (
                    ((29 / 10) - a) ** (3 / 7)))


def m5(a, b):
    return (-7) * (
            (
                    316200000.0 * b ** 5 + 1035300000.0 * b ** 4 + 3502765000.0 * b ** 3 + 12548140500.0 * b ** 2 + 50945450430.0 * b + 344730881243.0) * (
                    ((29 / 10) - b) ** (3 / 7)) - (
                    316200000.0 * a ** 5 + 1035300000.0 * a ** 4 + 3502765000.0 * a ** 3 + 12548140500.0 * a ** 2 + 50945450430.0 * a + 344730881243.0) * (
                    ((29 / 10) - a) ** (3 / 7))) * (1 / 12015600000.0)


def main():
    print('Поиск интеграла функции 4*cos(2.5*x)*exp((4*x)/7)+2.5*sin(5.5*x)*exp(-(3*x)/5)+4.3*x на интервале [1.8;2.9]')
    print()
    real = 57.48462064655285571820619434508191055583
    print(f'Точное значение интеграла: {real}')
    print()

    moments = np.array([
        m0, m1, m2
    ])
    value, error = newton_cots(F, 1.8, 2.9, moments)
    print(
        f'Значение интеграла с помощью метода Ньютона-Котса, найденное без поиска оптимального шага:'
        f' {value} c погрешностью {error} (истинная погрешность: {value - real})'
    )
    print()
    value, error = newton_cots(F, 1.8, 2.9, moments, optimal=True)
    print(
        f'Значение интеграла с помощью метода Ньютона-Котса, найденное c поиском оптимального шага:'
        f' {value} c погрешностью {error} (истинная погрешность: {value - real})'
    )

    print()
    moments = np.array([
        m0, m1, m2, m3, m4, m5
    ])
    value, error = gauss(F, 1.8, 2.9, moments)
    print(
        f'Значение интеграла с помощью метода Гаусса, найденное без поиска оптимального шага:'
        f' {value} c погрешностью {error} (истинная погрешность: {value - real})'
    )
    print()
    value, error = gauss(F, 1.8, 2.9, moments, optimal=True)
    print(
        f'Значение интеграла с помощью метода Гаусса, найденное c поиском оптимального шага: '
        f'{value} c погрешностью {error} (истинная погрешность: {value - real})')
    print()


if __name__ == '__main__':
    main()
