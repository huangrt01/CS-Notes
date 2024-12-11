# refer to: https://blog.csdn.net/little_kid_pea/article/details/79199090
# tensorboard --inspect --event_file=events.out.tfevents.1712654183.nXXX --tag=deep_part/model_part

import tensorflow as tf
import numpy

tf_event_file_path = "events.out.tfevents.1712680896.nXXX"
all_tensor = []
for e in tf.compat.v1.train.summary_iterator(tf_event_file_path):
    for v in e.summary.value:
        if v.tag == 'deep_part/model_part':
          fb = numpy.frombuffer(v.tensor.tensor_content, dtype=numpy.float32)
          all_tensor.append(fb)
numpy.set_printoptions(threshold=numpy.inf)
# 查看all_tensor中保存的值
print(all_tensor[0])