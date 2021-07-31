from typing import TypeVar, Type


class A:
    f1 = 0

    def __init__(self, param):
        print(f"{A.f1=}, {param=}")
        self.f1 = param
        print(f"{A.f1=}, {self.f1=}")
    T_ = TypeVar('T_', A, type(None))



a1 : A.T_ = 5 # A(3)
a2: A.T_ = None

print(f"outer {A.f1=}")
aa: A = A(5)
print(f"outer {A.f1=}")

# T = TypeVar('T')
# STRN_ = TypeVar('STRN_', str, bytes, None)
# s : STRN_ = "5"
# ss : STRN_ = None
