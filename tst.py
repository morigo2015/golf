

class AA:
    bb = "just str"
    cc = "cc"
    def __init__(self,par):
        self.par = par
        print(f"{par=} {AA.bb=} {self.bb=}")


class BB:
    c = 1


a1 = AA(10)
a2 = AA(20)
AA.bb = "new str"
a3 = AA(30)