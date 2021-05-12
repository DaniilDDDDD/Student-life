class Node:
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []
        self.child = []


class BTree:
    def __init__(self, t):
        self.root = Node(True)
        self.t = t  # порядок дерева


def main():
    with open('input.txt', 'r') as input_file:
        line = input_file.readline().split()
        n = int(line[0])

        line = input_file.readline().split()
        count = 1
        while line[0] == '-':
            line = input_file.readline().split()
            count += 1

        tree = BTree(float(line[1]))

        with open('output.txt', 'w') as output_file:
            output_file.write(str(+ '\n')

            for i in range(n - count):
                line = input_file.readline().split()
                if line[0] == '+':
                    tree.put(float(line[1]))
                    output_file.write(str(tree.balance_factor()) + '\n')
                elif line[0] == '-':
                    tree.remove(float(line[1]))
                    output_file.write(str(tree.balance_factor()) + '\n')
                elif line[0] == '?':
                    output_file.write(str(tree.contains(float(line[1]))) + '\n')


if __name__ == '__main__':
    main()
