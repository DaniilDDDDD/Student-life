import numpy as np


def lsd_sort(n, m, k, lines):
    lines = np.array(lines)

    arrange = np.arange(n)
    result = arrange.copy()
    array = arrange.copy()

    for j in reversed(range(m - k, m)):
        d = {}
        for i in range(97, 123):
            d[chr(i)] = 0
        for line in lines:
            d[line[j]] += 1

        count = 0
        for i in range(97, 123):
            if d[chr(i)] != 0:
                l = lines[:, j]
                indexes = np.where(l == chr(i))[0]
                for r in indexes:
                    array[r], array[count] = array[count], array[r]
                    count += 1
                lines = lines[array]
                result = result[array]
                array = arrange.copy()
                d[chr(i)] = 0

    return result


def main():
    with open('input.txt', 'r') as input_f:
        line = input_f.readline().split()
        n = int(line[0])
        m = int(line[1])
        k = int(line[2])
        lines = []
        for line in input_f:
            lines.append(list(line[:m]))

    out = lsd_sort(n, m, k, lines)

    with open('output.txt', 'w') as output_f:
        output_f.write(' '.join(map(str, out)))


if __name__ == '__main__':
    main()
