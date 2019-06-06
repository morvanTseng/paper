
def dec(f):
    def wrapper(a):
        print("this is a deccccc", a)
    return wrapper


@dec
def func(a):
    print("this is a func", a)

class ThisClass:
    pass


if __name__ == "__main__":
    t = ThisClass()
    print(ThisClass)
    print(type(int))
    print(type(ThisClass))
    print(type(t))