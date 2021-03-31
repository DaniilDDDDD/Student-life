class queue():
    queue = []
    
    def __init__(self, init_list=None):
        if init_list:
            self.add(init_list)
    
    def add(self, elem , many=False):
        if many:
            self.queue.extend(elem)
        else:
            self.queue.append(elem)
    
    def get(self):
        return self.queue.pop(0)
    
    def find_min(self):
        return min(self.queue)
    
    
    
def main():
    q = queue()
    mins = []
    n = int(input())
    for i in range(n):
        inp = input()
        if inp[0]=='+':
            q.add(float(inp[2:]))
        elif inp[0]=='-':
            q.get()
        elif inp[0]=='?':
            mins.append(q.find_min())
            
    for m in mins:
        print(m)
    
if __name__=='__main__':
    main()