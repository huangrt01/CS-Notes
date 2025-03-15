
class Circle:
    class_attribute = 10  # 类属性，被共享

    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value


    _instance = None

    @staticmethod
    def getInstance(*args, **kwargs):
        if Circle._instance is None:
            Circle._instance = Circle(*args, **kwargs)
        return Circle._instance


class Singleton(type):
  _instances: Dict[type, "Singleton"] = {}

  def __call__(cls, *args, **kwargs):
    if cls not in cls._instances:
      cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
    return cls._instances[cls]

  def instance(cls: Any, *args: Any, **kwargs: Any) -> "Singleton":
    return cls(*args, **kwargs)
