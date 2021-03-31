def prepare_killer(length):
    sequence = [x for x in range(length)]
    r = length - 1
    for l in range(r):
        p = (l + r) // 2
        sequence[l],sequence[p] = sequence[p], sequence[l]
    return sequence


def main():
    n = int(input())
    print(prepare_killer(n))


if __name__=='__main__':
    main()