class BTreeSet:

    def __init__(self, degree):
        self.minkeys = degree - 1
        self.maxkeys = degree * 2 - 1
        self.root = BTreeSet.Node(self.maxkeys, True)

    def add(self, obj):
        root = self.root
        if len(root.keys) == self.maxkeys:
            child = root
            self.root = root = BTreeSet.Node(self.maxkeys, False)  # Increment tree height
            root.children.append(child)
            root.split_child(self.minkeys, self.maxkeys, 0)

        node = root
        while True:
            assert len(node.keys) < self.maxkeys
            assert node is root or len(node.keys) >= self.minkeys
            found, index = node.search(obj)
            if found:
                return

            if node.is_leaf():
                node.keys.insert(index, obj)
                return

            else:
                child = node.children[index]
                if len(child.keys) == self.maxkeys:
                    node.split_child(self.minkeys, self.maxkeys, index)
                    if obj == node.keys[index]:
                        return
                    elif obj > node.keys[index]:
                        child = node.children[index + 1]
                node = child

    def exists(self, key, root=None):
        if root is None:
            root = self.root

        if root.search(key)[0]:
            return True
        if root.children:
            for child in root.children:
                if self.exists(key, child):
                    return True
        return False

    def remove(self, obj):
        if not self._remove(obj):
            raise KeyError(str(obj))

    def discard(self, obj):
        self._remove(obj)

    def _remove(self, obj):
        root = self.root
        found, index = root.search(obj)
        node = root
        while True:
            assert len(node.keys) <= self.maxkeys
            assert node is root or len(node.keys) > self.minkeys
            if node.is_leaf():
                if found:
                    node.remove_key(index)
                return found

            else:
                if found:
                    left, right = node.children[index: index + 2]
                    if len(left.keys) > self.minkeys:
                        node.keys[index] = left.remove_max(self.minkeys)
                        return True
                    elif len(right.keys) > self.minkeys:
                        node.keys[index] = right.remove_min(self.minkeys)
                        return True
                    else:
                        node.merge_children(self.minkeys, index)
                        if node is root and len(root.keys) == 0:
                            assert len(root.children) == 1
                            self.root = root = left
                        node = left
                        index = self.minkeys

                else:
                    child = node.ensure_child_remove(self.minkeys, index)
                    if node is root and len(root.keys) == 0:
                        assert len(root.children) == 1
                        self.root = root = root.children[0]
                    node = child
                    found, index = node.search(obj)

    def __iter__(self):
        stack = []

        def push_left_path(node):
            while True:
                stack.append((node, 0))
                if node.is_leaf():
                    break
                node = node.children[0]

        push_left_path(self.root)

        while len(stack) > 0:
            node, index = stack.pop()
            if node.is_leaf():
                assert index == 0
                yield from node.keys
            else:
                yield node.keys[index]
                index += 1
                if index < len(node.keys):
                    stack.append((node, index))
                push_left_path(node.children[index])

    class Node:

        def __init__(self, maxkeys, leaf):
            self.keys = []
            self.children = None if leaf else []

        def is_leaf(self):
            return self.children is None

        def search(self, obj):
            keys = self.keys
            i = 0
            while i < len(keys):
                if obj == keys[i]:
                    return True, i
                elif obj > keys[i]:
                    i += 1
                else:
                    break
            return False, i

        def split_child(self, minkeys, maxkeys, index):
            left = self.children[index]
            right = BTreeSet.Node(maxkeys, left.is_leaf())
            self.children.insert(index + 1, right)

            if not left.is_leaf():
                right.children.extend(left.children[minkeys + 1:])
                del left.children[minkeys + 1:]

            self.keys.insert(index, left.keys[minkeys])
            right.keys.extend(left.keys[minkeys + 1:])
            del left.keys[minkeys:]

        def ensure_child_remove(self, minkeys, index):
            child = self.children[index]
            if len(child.keys) > minkeys:
                return child

            left = self.children[index - 1] if index >= 1 else None
            right = self.children[index + 1] if index < len(self.keys) else None
            internal = not child.is_leaf()

            if left is not None and len(left.keys) > minkeys:
                if internal:
                    child.children.insert(0, left.children.pop(-1))
                child.keys.insert(0, self.keys[index - 1])
                self.keys[index - 1] = left.remove_key(len(left.keys) - 1)
                return child
            elif right is not None and len(right.keys) > minkeys:
                if internal:
                    child.children.append(right.children.pop(0))
                child.keys.append(self.keys[index])
                self.keys[index] = right.remove_key(0)
                return child
            elif left is not None:
                self.merge_children(minkeys, index - 1)
                return left
            elif right is not None:
                self.merge_children(minkeys, index)
                return child
            else:
                raise AssertionError("Impossible condition")

        def merge_children(self, minkeys, index):
            left, right = self.children[index: index + 2]
            if not left.is_leaf():
                left.children.extend(right.children)
            del self.children[index + 1]
            left.keys.append(self.remove_key(index))
            left.keys.extend(right.keys)

        def remove_min(self, minkeys):
            node = self
            while True:
                if node.is_leaf():
                    return node.remove_key(0)
                else:
                    node = node.ensure_child_remove(minkeys, 0)

        def remove_max(self, minkeys):
            node = self
            while True:
                if node.is_leaf():
                    return node.remove_key(len(node.keys) - 1)
                else:
                    node = node.ensure_child_remove(minkeys, len(node.children) - 1)

        def remove_key(self, index):
            return self.keys.pop(index)


def main():
    with open('input.txt', 'r') as input_file:
        line = input_file.readline().split()
        n = int(line[0])

        tree = BTreeSet(3)

        with open('output.txt', 'w') as output_file:

            for i in range(n):
                line = input_file.readline().split()
                if line[0] == '+':
                    tree.add(float(line[1]))
                    length = len(tree.root.keys)
                    keys = " ".join(map(str, tree.root.keys))
                    output_file.write(f'{length}: ' + keys + '\n')
                elif line[0] == '-':
                    try:
                        tree.remove(float(line[1]))
                        length = len(tree.root.keys)
                        keys = " ".join(map(str, tree.root.keys))
                        output_file.write(f'{length}: ' + keys + '\n')
                    except KeyError:
                        output_file.write('Key does not exist!' + '\n')
                elif line[0] == '?':
                    output_file.write(str(tree.exists(float(line[1]))) + '\n')


if __name__ == '__main__':
    main()
