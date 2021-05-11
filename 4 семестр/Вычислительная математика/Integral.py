import math
import numpy as np
from scipy import integrate


def integral(f, a, b, p, p_args, moments, gauss=False, optimal=False, step_number=2, r=3, error=10 ** (-6)):
    if not gauss:  # метод Ньютона-Котса
        x = np.array([a, (a + b) / 2, b], dtype=float)
        xes = np.array([x ** i for i in range(len(x))], dtype=float)

        A = np.linalg.solve(xes, moments.transpose())

    else:  # метод Гаусса
        temp_system = np.array(
            [[moments[i + j] for j in range(int((len(moments) + 1) / 2))] for i in range(int((len(moments) + 1) / 2))],
            dtype=float
        )
        temp_b = np.array(
            [-moments[int((len(moments) + 1) / 2) + s] for s in range(int((len(moments) + 1) / 2))],
            dtype=float
        )
        w = np.linalg.solve(temp_system, temp_b.transpose())
        w_roots = np.roots(w)

        xes = np.array([w_roots ** i for i in range(len(w_roots))], dtype=float)
        A = np.linalg.solve(xes, moments[:int((len(moments) + 1) / 2)])

    # модуль 3ей производной заданной функции f(x) < 360 на [1.8, 2.9]
    e = 0
    # e = (360 / 6) * (
    #     integrate.quad(
    #         lambda y: abs(p(y, a, b, p_args[0], p_args[1]) * ((x - x[0]) * (x - x[1]) * (x - x[2]))),
    #         1.8,
    #         2.9
    #     )[0]
    # )

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
            current_interval = [x[i], (x[i + 1] + x[i]) / 2, x[i + 1]]
            for m in range(len(current_interval)):
                s += A[m] * f(current_interval[m])
        S = np.append(S, s)

    # Метод Эйткена
    m = int((- math.log((S[2] - S[1]) / (S[1] - S[0]))) / math.log(L))

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

    return S[-1], R[-1], e


def P(x, a, b, alpha, beta):
    return ((x - a) ** (-alpha)) * ((b - x) ** (-beta))


def F(x):
    return 4 * math.cos(2.5 * x) * math.exp((4 * x) / 7) + 2.5 * math.sin(5.5 * x) * math.exp(-(3 * x) / 5) + 4.3 * x


def main():
    print('Поиск интеграла функции 4*cos(2.5*x)*exp((4*x)/7)+2.5*sin(5.5*x)*exp(-(3*x)/5)+4.3*x на интервале [1.8;2.9]')
    print()
    value = integrate.quad(F, 1.8, 2.9)
    print(f'Значение интеграла: {value[0]} c погрешностью {value[1]}')
    print()

    moments = np.array([
        2.43062, 6.24668, 16.3083
    ])
    value, error, methodical_error = integral(F, 1.8, 2.9, P, (0, 4 / 7), moments)
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
