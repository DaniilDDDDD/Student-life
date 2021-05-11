class Node:

    def __init__(self, k):
        self.key = k
        self.height = 1
        self.left = None
        self.right = None

    def __str__(self):
        return str(self.key)

    def __repr__(self):
        return str(self.key)


class AVLTree:

    def __init__(self, key):
        self.root_node = Node(key)

    def __height(self, node) -> int:
        if node is None:
            return 0
        return node.height

    def balance_factor(self, node=None) -> int:
        if node is None:
            node = self.root_node
        if node is None:
            return 0

        return self.__height(node.right) - self.__height(node.left)

    def __fix_hight(self, node):
        h_l = self.__height(node.left)
        h_r = self.__height(node.right)
        if h_l > h_r:
            node.height = h_l + 1
        else:
            node.height = h_r + 1

    def __rotate_right(self, p):
        q = p.left
        p.left = q.right
        q.right = p
        self.__fix_hight(p)
        self.__fix_hight(q)
        return q

    def __rotate_left(self, q):
        p = q.right
        q.right = p.left
        p.left = q
        self.__fix_hight(q)
        self.__fix_hight(p)
        return p

    def __balance(self, p):
        self.__fix_hight(p)
        if self.balance_factor(p) == 2:
            if self.balance_factor(p.right) < 0:
                p.right = self.__rotate_right(p.right)
            return self.__rotate_left(p)
        if self.balance_factor(p) == -2:
            if self.balance_factor(p.left) > 0:
                p.left = self.__rotate_left(p.left)
            return self.__rotate_right(p)
        return p

    def __insert(self, node, key):
        if node is None:
            return Node(key)
        if key < node.key:
            node.left = self.__insert(node.left, key)
        else:
            node.right = self.__insert(node.right, key)
        return self.__balance(node)

    def put(self, key):
        self.root_node = self.__insert(self.root_node, key)

    def __find_min(self, node):
        if node.left is None:
            return node
        return self.__find_min(node.left)

    def __removemin(self, node):
        if node.left is None:
            return node.right
        node.left = self.__removemin(node.left)
        return self.__balance(node)

    def __rem(self, key, node):

        if node is None:
            return None
        if key < node.key:
            node.left = self.__rem(key, node.left)
        elif key > node.key:
            node.right = self.__rem(key, node.right)
        else:
            q = node.left
            r = node.right
            if r is None:
                return q
            m = self.__find_min(r)
            m.right = self.__removemin(r)
            m.left = q
            return self.__balance(m)
        return self.__balance(node)

    def remove(self, key):
        self.root_node = self.__rem(key, self.root_node)

    def contains(self, key, node=None):
        if node is None:
            if self.root_node is None:
                return False
            node = self.root_node

        if key == node.key:
            return True
        elif key < node.key:
            if node.left is None:
                return False
            return self.contains(key, node.left)
        else:
            if node.right is None:
                return False
            return self.contains(key, node.right)


def main():
    with open('input.txt', 'r') as input_file:
        line = input_file.readline().split()
        n = int(line[0])

        line = input_file.readline().split()
        count = 1
        while line[0] == '-':
            line = input_file.readline().split()
            count += 1
        tree = AVLTree(float(line[1]))

        with open('output.txt', 'w') as output_file:
            output_file.write(str(tree.balance_factor()) + '\n')

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
