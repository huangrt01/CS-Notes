import time
import numpy as np

start = time.time()
weights = {"Variable{}".format(i): np.random.normal(size=5000000)
           for i in range(10)}
end = time.time()
print ("initialization :{}".format(end-start))

import pickle
start = time.time()
# Serialize the weights with pickle. Then deserialize them.
pickled_weights = pickle.dumps(weights)   
new_weights = pickle.loads(pickled_weights)
end = time.time()
print ("serialize and deserialize with pickle :{}".format(end-start))

import ray
ray.init()
# Serialize the weights & copyinto object store. Then deserialize them.
start = time.time()
weights_id = ray.put(weights)   
new_weights = ray.get(weights_id)
end = time.time()
print("serialize and deserialize with ray :{}".format(end-start))
