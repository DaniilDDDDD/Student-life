class Node:
    def __init__(self, leaf=False, values=None):
        self.leaf = leaf
        self.keys = []
        self.child = []
        if values is not None:
            self.values = values
        else:
            self.values = [values] * len(self.keys)


class BTree:
    def __init__(self, t):
        self.root = Node(True)
        self.t = t  # порядок дерева

    # Search key
    def search_key(self, k, x=None):
        if x is not None:
            i = 0
            while i < len(x.keys) and k > x.keys[i]:
                i += 1
            if i < len(x.keys) and k == x.keys[i]:
                return x, i
            elif x.leaf:
                return None
            else:
                return self.search_key(k, x.child[i])
        else:
            return self.search_key(k, self.root)

    # Insert the key
    def insert(self, k, value):
        root = self.root
        # если self.root заполнен, то создаётся новый корень с ключём равным среднему элементу изначального self.root
        if len(root.keys) == (2 * self.t) - 1:
            temp = Node()
            self.root = temp
            temp.child.insert(0, root)
            self.split(temp, 0)
            self.insert_non_full(temp, k, value)
        # если self.root не заполнен, вставляем элементы так же, как и в любой другой узел
        else:
            self.insert_non_full(root, k, value)

    # Insert non full condition
    def insert_non_full(self, x, k, value):
        i = len(x.keys) - 1
        if x.leaf:
            x.keys.append(None)
            x.values.append(None)
            while i >= 0 and k < x.keys[i]:
                x.keys[i + 1] = x.keys[i]
                x.values[i + 1] = x.values[i]
                i -= 1
            x.keys[i + 1] = k
            x.values[i + 1] = value
        else:
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            if len(x.child[i].keys) == (2 * self.t) - 1:
                self.split(x, i)
                if k > x.keys[i]:
                    i += 1
            self.insert_non_full(x.child[i], k, value)

    # Split
    def split(self, x, i):
        t = self.t
        y = x.child[i]
        z = Node(y.leaf)
        x.child.insert(i + 1, z)
        x.keys.insert(i, y.keys[t - 1])
        x.values.insert(i, y.values[t - 1])
        z.keys = y.keys[t: (2 * t) - 1]
        z.values = y.values[t: (2 * t) - 1]
        y.keys = y.keys[0: t - 1]
        y.values = y.values[0: t - 1]
        if not y.leaf:
            z.child = y.child[t: 2 * t]
            y.child = y.child[0: t - 1]

    # Delete a node
    def delete(self, x, k):
        t = self.t
        i = 0
        while i < len(x.keys) and k > x.keys[i]:
            i += 1
        if x.leaf:
            if i < len(x.keys) and x.keys[i] == k:
                x.keys.pop(i)
                x.values.pop(i)
                return
            return

        if i < len(x.keys) and x.keys[i] == k:
            return self.delete_internal_node(x, k, i)
        elif len(x.child[i].keys) >= t:
            self.delete(x.child[i], k)
        else:
            if i != 0 and i + 2 < len(x.child):
                if len(x.child[i - 1].keys) >= t:
                    self.delete_sibling(x, i, i - 1)
                elif len(x.child[i + 1].keys) >= t:
                    self.delete_sibling(x, i, i + 1)
                else:
                    self.delete_merge(x, i, i + 1)
            elif i == 0:
                if len(x.child[i + 1].keys) >= t:
                    self.delete_sibling(x, i, i + 1)
                else:
                    self.delete_merge(x, i, i + 1)
            elif i + 1 == len(x.child):
                if len(x.child[i - 1].keys) >= t:
                    self.delete_sibling(x, i, i - 1)
                else:
                    self.delete_merge(x, i, i - 1)
            self.delete(x.child[i], k)

    # Delete internal node
    def delete_internal_node(self, x, k, i):
        t = self.t
        if x.leaf:
            if x.keys[i] == k:
                x.keys.pop(i)
                x.values.pop(i)
                return
            return

        if len(x.child[i].keys) >= t:
            x.keys[i], x.values[i] = self.delete_predecessor(x.child[i])
            return
        elif len(x.child[i + 1].keys) >= t:
            x.keys[i], x.values[i] = self.delete_successor(x.child[i + 1])
            return
        else:
            self.delete_merge(x, i, i + 1)
            self.delete_internal_node(x.child[i], k, self.t - 1)

    # Delete the predecessor
    def delete_predecessor(self, x):
        if x.leaf:
            return x.keys.pop(), x.values.pop()
        n = len(x.keys) - 1
        if len(x.child[n].keys) >= self.t:
            self.delete_sibling(x, n + 1, n)
        else:
            self.delete_merge(x, n, n + 1)
        self.delete_predecessor(x.child[n])

    # Delete the successor
    def delete_successor(self, x):
        if x.leaf:
            return x.keys.pop(0), x.values.pop(0)
        if len(x.child[1].keys) >= self.t:
            self.delete_sibling(x, 0, 1)
        else:
            self.delete_merge(x, 0, 1)
        self.delete_successor(x.child[0])

    # Delete resolution
    def delete_merge(self, x, i, j):
        cnode = x.child[i]

        if j > i:
            rsnode = x.child[j]
            cnode.keys.append(x.keys[i])
            for k in range(len(rsnode.keys)):
                cnode.keys.append(rsnode.keys[k])
                if len(rsnode.child) > 0:
                    cnode.child.append(rsnode.child[k])
            if len(rsnode.child) > 0:
                cnode.child.append(rsnode.child.pop())
            new = cnode
            x.keys.pop(i)
            x.values.pop(i)
            x.child.pop(j)
        else:
            lsnode = x.child[j]
            lsnode.keys.append(x.keys[j])
            for i in range(len(cnode.keys)):
                lsnode.keys.append(cnode.keys[i])
                if len(lsnode.child) > 0:
                    lsnode.child.append(cnode.child[i])
            if len(lsnode.child) > 0:
                lsnode.child.append(cnode.child.pop())
            new = lsnode
            x.keys.pop(j)
            x.values.pop(j)
            x.child.pop(i)

        if x == self.root and len(x.keys) == 0:
            self.root = new

    # Delete the sibling
    def delete_sibling(self, x, i, j):
        cnode = x.child[i]
        if i < j:
            rsnode = x.child[j]
            cnode.keys.append(x.keys[i])
            cnode.values.append(x.values[i])
            x.keys[i] = rsnode.keys[0]
            x.values[i] = rsnode.values[0]
            if len(rsnode.child) > 0:
                cnode.child.append(rsnode.child[0])
                rsnode.child.pop(0)
            rsnode.keys.pop(0)
            rsnode.values.pop(0)
        else:
            lsnode = x.child[j]
            cnode.keys.insert(0, x.keys[i - 1])
            cnode.values.insert(0, x.values[i - 1])
            x.keys[i - 1] = lsnode.keys.pop()
            x.values[i - 1] = lsnode.values.pop()
            if len(lsnode.child) > 0:
                cnode.child.insert(0, lsnode.child.pop())


def main():

    B = BTree(3)

    for i in range(10):
        B.insert(i, 2 * i)
        print(B.root.keys)

    B.delete(B.root, 3)

    print('sdfas')



    # with open('input.txt', 'r') as input_file:
    #     line = input_file.readline().split()
    #     n = int(line[0])
    #
    #     line = input_file.readline().split()
    #     count = 1
    #     while line[0] == '-':
    #         line = input_file.readline().split()
    #         count += 1
    #
    #     tree = BTree(float(line[1]))
    #
    #     with open('output.txt', 'w') as output_file:
    #         output_file.write(str(' ' + '\n')
    #
    #                           for i in range(n - count):
    #         line = input_file.readline().split()
    #         if line[0] == '+':
    #             tree.put(float(line[1]))
    #         output_file.write(str(tree.balance_factor()) + '\n')
    #         elif line[0] == '-':
    #         tree.remove(float(line[1]))
    #         output_file.write(str(tree.balance_factor()) + '\n')
    #         elif line[0] == '?':
    #         output_file.write(str(tree.contains(float(line[1]))) + '\n')

if __name__ == '__main__':
    main()
