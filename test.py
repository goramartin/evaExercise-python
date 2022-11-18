class Mutation:

    def __init__(self, step_size, fitness):
        self.step_size = step_size
        self.fit = fitness

    def __call__(self, ind):
        print(11)

    def update_step_size():
        pass

class MutationX(Mutation):

    def __init__(self, step_size, fitness):
        super().__init__(step_size, fitness)
        x = 1

    def __call__(self, ind):
        print(22)

    def update_step_size():
        pass


xx = MutationX(44, 2)
print(xx.step_size)