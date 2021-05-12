import math
import numpy as np
from scipy import integrate


def newton_cots(f, a, b, p, p_args, moments, optimal=False, step_number=2, r=3,
                error=10 ** (-6)):  # метод Ньютона-Котса
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
            for m in range(len(current_interval)):
                s += A[m] * f(current_interval[m])
        S = np.append(S, s)

    # Метод Эйткена
    m = (- math.log((S[2] - S[1]) / (S[1] - S[0]))) / math.log(L)

    H = []
    for h in H_begin:
        H = np.append(H, [h ** i for i in range(m, m + len(H_begin))])
    H = np.array(H, dtype=float)

    R = np.linalg.solve(H, S.transpose())

    if not optimal:  # без поиска оптимального шага h
        while abs(R[-1]) >= error:
            h = H_begin[-1] / L
            H_begin = np.append(H_begin, h)
            x = np.arange(a, b, h)
            x = np.append(x, b)
            s = 0
            for i in range(len(x) - 1):
                current_interval = [x[i], (x[i + 1] + x[i]) / 2, x[i]]
                for k in range(len(current_interval)):
                    s += A[k] * f(current_interval[m])
            S = np.append(S, s)

            m = int((- math.log((S[-1] - S[-2]) / (S[-2] - S[-3]))) / math.log(L))

            H = []
            for h in H_begin[-r:]:
                H = np.append(H, [h ** i for i in range(m, m + r)])
            H = np.array(H, dtype=float)

            R = np.linalg.solve(H, S[-r:].transpose())

    else:  # с поиском оптимального шага h
        h = H_begin[0] * ((error / abs(R[0])) ** (1 / m))
        H_begin = np.append(H_begin, h)
        x = np.arange(a, b, h)
        x = np.append(x, b)
        s = 0
        for i in range(len(x) - 1):
            current_interval = [x[i], (x[i + 1] + x[i]) / 2, x[i]]
            for k in range(len(current_interval)):
                s += A[k] * f(current_interval[m])
        S = np.append(S, s)

        m = int((- math.log((S[-1] - S[-2]) / (S[-2] - S[-3]))) / math.log(L))

        H = []
        for h in H_begin[-r:]:
            H = np.append(H, [h ** i for i in range(m, m + r)])
        H = np.array(H, dtype=float)

        R = np.linalg.solve(H, S[-r:].transpose())

    return S[-1], R[-1]


def P(x, a, b, alpha, beta):
    return ((x - a) ** (-alpha)) * ((b - x) ** (-beta))


def F(x):
    return 4 * math.cos(2.5 * x) * math.exp((4 * x) / 7) + 2.5 * math.sin(5.5 * x) * math.exp(-(3 * x) / 5) + 4.3 * x


def m0(a, b):
    return (-2 + 1 / 3) * (((2.9 - b) ** (3 / 7)) - ((2.9 - a) ** (3 / 7)))


def m1(a, b):
    return (-0.7) * ((b + 6.76667) * ((2.9 - b) ** (3 / 7)) - (a + 6.76667) * ((2.9 - a) ** (3 / 7)))


def m2(a, b):
    return (-0.411765) * ((b ** 2 + 4.06 * b + 27.4727) * ((2.9 - b) ** (3 / 7)) - (a ** 2 + 4.06 * a + 27.4727) * (
            (2.9 - a) ** (3 / 7)))


def m3(a, b):
    return (-0.291667) * ((b ** 3 + 3.58235 * b * b + 14.5444 * b + 98.4168) * ((2.9 - b) ** (3 / 7)) - (
            a ** 3 + 3.58235 * a * a + 14.5444 * a + 98.4168) * ((2.9 - a) ** (3 / 7)))


def m4(a, b):
    return (-0.225806) * (
            (b ** 4 + 3.38333 * (b ** 3) + 12.1203 * (b ** 2) + 49.2084 * b + 332.977) * ((2.9 - b) ** (3 / 7)) - (
            a ** 4 + 3.38333 * (a ** 3) + 12.1203 * (a ** 2) + 49.2084 * a + 332.977) * ((2.9 - a) ** (3 / 7)))


def m5(a, b):
    return (-0.184211) * (
            (b ** 5 + 3.27419 * (b ** 4) + 11.0777 * (b ** 3) + 39.6842 * (b * 2) + 161.118 * b + 1090.23) * (
            (2.9 - b) ** (3 / 7)) - (a ** 5 + 3.27419 * (a ** 4) + 11.0777 * (a ** 3) + 39.6842 * (
            a * 2) + 161.118 * a + 1090.23) * ((2.9 - a) ** (3 / 7)))


def main():
    print('Поиск интеграла функции 4*cos(2.5*x)*exp((4*x)/7)+2.5*sin(5.5*x)*exp(-(3*x)/5)+4.3*x на интервале [1.8;2.9]')
    print()
    value = integrate.quad(F, 1.8, 2.9)
    print(f'Значение интеграла: {value[0]} c погрешностью {value[1]}')
    print()

    moments = np.array([
        m0, m1, m2
    ])
    value, error, methodical_error = newton_cots(F, 1.8, 2.9, P, (0, 4 / 7), moments)
    print(
        f'Значение интеграла с помощью метода Ньютона-Котса, найденное без поиска оптимального шага:'
        f' {value} c погрешностью {error} и методической погрешностью {methodical_error}'
    )
    print()
    value, error, methodical_error = integral(F, 1.8, 2.9, P, (0, 4 / 7), moments, optimal=True)
    print(
        f'Значение интеграла с помощью метода Ньютона-Котса, найденное c поиском оптимального шага:'
        f' {value} c погрешностью {error} и методической погрешностью {methodical_error}'
    )
    print()
    moments = np.array([
        2.43062, 6.24668, 16.3083, 43.1542, 115.505, 312.147
    ])
    value, error, methodical_error = integral(F, 1.8, 2.9, P, (0, 4 / 7), moments, gauss=True)
    print(
        f'Значение интеграла с помощью метода Гаусса, найденное без поиска оптимального шага:'
        f' {value} c погрешностью {error} и методической погрешностью {methodical_error}'
    )
    print()
    value, error, methodical_error = integral(F, 1.8, 2.9, P, (0, 4 / 7), moments, gauss=True, optimal=True)
    print(
        f'Значение интеграла с помощью метода Гаусса, найденное c поиском оптимального шага:'
        f' {value} c погрешностью {error} и методической погрешностью {methodical_error}'
    )
    print()


if __name__ == '__main__':
    main()
