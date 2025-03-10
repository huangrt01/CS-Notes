### GIL

https://wiki.python.org/moin/GlobalInterpreterLock


### Decorator

def test_task_decorator(*args, task_filter: Union[str, List[str]] = None, **kwargs):
  def decorator(func: Callable):
    def wrapper(self, *extra_args, **extra_kwargs):
      try:
        for name, t in test_task_store.get_tasks().items():
          if task_filter is None or (isinstance(task_filter, str) and name == task_filter) or (
              isinstance(task_filter, list) and name in task_filter):
            try:
              func(self, *extra_args, task=t, task_name=name, **extra_kwargs, **kwargs)
            except Exception as e:
              logging.error(f"Error occurred while running test for task {name}: {e}", exc_info=True)
      except Exception as e:
        logging.error(f"Error in test decorator: {e}", exc_info=True)

    return wrapper

  return decorator