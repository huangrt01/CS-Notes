
func.__get__(opt, opt.__class__)(*args, **kwargs)


### Monkey Patch

可以修改类和模块
math.sqrt = new_sqrt