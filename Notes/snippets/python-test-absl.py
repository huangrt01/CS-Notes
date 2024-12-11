import tensorflow as tf
from absl.testing import parameterized

class MyOpTest(tf.test.TestCase, parameterized.TestCase):
	@classmethod
  	def setUpClass(cls):
  		...

  	@parameterized.parameters([(MyType.EXAMPLE,), (PbType.INSTANCE,)])
  	def test_basic(self, type: MyType):
  		...