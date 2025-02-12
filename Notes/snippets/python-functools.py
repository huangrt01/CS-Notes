
import functools

def my_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print("Before function call")
        result = func(*args, **kwargs)
        print("After function call")
        return result
    return wrapper

@my_decorator
def add(a, b):
    """This function adds two numbers."""
    return a + b

print(add.__name__)  # 输出 'add'
print(add.__doc__)   # 输出 'This function adds two numbers.'
