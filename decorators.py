def decarator1(func):
    print("stmt1")
    def wrapper():
        print("stmt2")
        print("before func call")
        print("stmt3")
        func()
        print("stmt4")
        print("after func call")
        print("stmt5")
    return wrapper


@decarator1
def print_name():
    print("stmt6")
    print('vamshi')
    print("stmt7")

print_name()
# print_name=decarator1(print_name)
# print_name()