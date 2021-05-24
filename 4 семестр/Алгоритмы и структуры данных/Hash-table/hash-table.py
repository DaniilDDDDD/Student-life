class Node:
    def __init__(self, data=None):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def contains(self, data):
        last = self.head
        while last:
            if data == last.data:
                return True
            else:
                last = last.next
        return False

    def add(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
            return
        last = self.head
        while last.next:
            last = last.next
        last.next = new_node

    def remove(self, data):
        head = self.head
        if head is not None:
            if head.data == data:
                self.head = head.next
                del head
                return
        last = head
        while head is not None:
            if head.data == data:
                break
            last = head
            head = head.next
        if head is None:
            return
        last.next = head.next
        del head

    def delete_every(self, data):
        nodes = []
        cur = self.head
        while cur:
            if cur.data == data:
                nodes.append(cur)
            cur = cur.next
        for node in nodes:
            self.remove(node.data)


class HashTable:
    def __init__(self):
        self.m = 8192
        self.table = [LinkedList() for x in range(self.m)]

    def __hash(self, key):
        return int(key % self.m)

    def add(self, item):
        self.table[self.__hash(item)].add(item)

    def delete(self, item):
        self.table[self.__hash(item)].delete_every(item)

    def contains(self, item):
        return self.table[self.__hash(item)].contains(item)


def main():
    with open('input.txt', 'r') as input_file:
        line = input_file.readline().split()
        n = int(line[0])
        h = HashTable()
        with open('output.txt', 'w') as output_file:
            for i in range(n):
                line = input_file.readline().split()
                if line[0] == '+':
                    h.add(float(line[1]))
                elif line[0] == '-':
                    h.delete(float(line[1]))
                elif line[0] == '?':
                    output_file.write(str(h.contains(float(line[1]))) + '\n')


if __name__ == '__main__':
    main()
