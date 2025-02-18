- unittest.mock

coverage run --source=monotorch -m pytest tests/
coverage html --show-contexts


- @pytest.mark.benchmark



import shutil
import tempfile



class Test(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test1(self):
    		new_ckpt_dir = f'{self.temp_dir}/new_saved'
    		...