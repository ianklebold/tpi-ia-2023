# When the args starting with one * whe can provide multiple parameters
# When the args starting with two * whe can provide multiple keywords and values (parameters)


# In the background the args are become in a tuple
def multiply(*args):
    result = 1
    for i in args:
        result *= i
    return result


# In the background the kargs are become in a dictionary
def total(**kwargs):
    print(type(kwargs))
    # Destructuration
    for key, value in kwargs.items():
        print(key, value)


def totalSum(n, **kwargs):
    power = 1
    sm = 0
    for i in range(n):
        power *= kwargs['a']
        power += kwargs['b']
    return power, sm


multiply(1, 2, 3, 4, 5, 6)
total(a=1, b=2, c=3, d=4)
print(type(totalSum(5, a=1, b=2))) #Return a tuple



class Phone:
    def __init__(self, **kw):
        self.make = kw.get('make')
        self.model = kw.get('model')

# .get in dict is like a optional in java, return none is null o the value

my_phone = Phone(make= "Apple")
print(my_phone.model)







