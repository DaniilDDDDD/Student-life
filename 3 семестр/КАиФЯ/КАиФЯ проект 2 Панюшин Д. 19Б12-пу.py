class Automat():
    # мертвое состояние (всегда последнее)
    states = [
        ('a', 'b', 'x'),
        {'x':(2, ('',), 'm'), 'a': (3, ('',), ''), 'b': (9, ('',), '')},
        {'x':(2, ('m',), 'm'), 'a': (3, ('m',), ''), 'b': (9, ('m',), '')},
        {'x':(9, ('m',''), ''), 'a': (4, ('m','n',''), 'n'), 'b': (9, ('m','n',''), '')},
        {'x':(9, ('m','n',''), ''), 'a': (3, ('n',), ''), 'b': (5, ('n',), '')},
        {'x':(9, ('n',), ''), 'a': (9, ('n',), ''), 'b': (6, ('n',), '')},
        {'x':(9, ('n',), ''), 'a': (9, ('n',), ''), 'b': (7, ('n',), 'd')},
        {'x':(8, ('m',), 'd'), 'a': (9, ('m','n'), ''), 'b': (5, ('n',), '')},
        {'x':(9, ('m',), 'd'), 'a': (9, ('m',), ''), 'b': (9, ('m',), '')},
        {'x':(9, ('n','m',''), 'd'), 'a': (9, ('n','m',''), 'd'), 'b': (9, ('n','m',''), 'd')},  
    ]
    stack = ''  # начальный символ
    
    def __init__(self, final_states, current_state=1, states=None):
        self.final_states = final_states
        self.current_state = current_state
        if states is not None:
            self.set_states(states)
        
    def set_states(self, states):
        self.states = states
        
    def add(self, string):
        self.stack = self.stack + string
    def pop(self):
        self.stack = self.stack[0:-1]
    def check(self):
        if self.stack == '':
            return ''
        return self.stack[-1]
    
    def read(self, char):
        if char not in self.states[0]:  # если символ не в алфавите
            self.current_state = self.states[-1][self.states[0][0]][0]
        state = self.states[self.current_state][char]
        if self.check() in state[1]:
            if state[2]=='d':  # символ d означает удаление верхнего символа стека
                self.pop()
            else:
                self.add(state[2])
            self.current_state = state[0]
        else:  # если нет совпадения верхнего символа стека с предполагаемыми на этом шаге 
            self.current_state = self.states[-1][self.states[0][0]][0]
        print(state)
        print(self.stack)
        return self.current_state  
    
    def recognize(self, string):
        if string=='':
            return True
        for char in string:
            self.read(char)
        if self.current_state in self.final_states and self.stack=='':
            print(self.current_state)
            return True
        return False