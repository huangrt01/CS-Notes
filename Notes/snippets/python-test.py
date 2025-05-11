### Basic

- unittest.mock

coverage run --source=monotorch -m pytest tests/
coverage html --show-contexts

### unittest

import shutil
import tempfile



class Test(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    @classmethod
    def setUpClass(cls):
        cls.xx = XX()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test1(self):
            new_ckpt_dir = f'{self.temp_dir}/new_saved'
            ...

### pytest

- @pytest.mark.benchmark

pytest.main(["-x", __file__])


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

### parametrized

- absl

from absl.testing import parameterized

class MyOpTest(parameterized.TestCase):
    @classmethod
    def setUpClass(cls):
        ...

    @parameterized.parameters(
        (1,2),
        (1,3),
        (1,10)
    )
    @test_task_decorator()
    def test_loader(self, N, M, task, task_name):
        ...


- pytest

@pytest.mark.parametrize("Z, H, N_CTX, HEAD_DIM", [(1, 2, 1024, 64)])
@pytest.mark.parametrize("causal", [True])