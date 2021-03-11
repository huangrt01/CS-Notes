def create_keras_model():
    from tensorflow import keras
    from tensorflow.keras import layers
    model = keras.Sequential()
    # Adds a densely-connected layer with 64 units to the model:
    model.add(layers.Dense(64, activation="relu", input_shape=(32, )))
    # Add another:
    model.add(layers.Dense(64, activation="relu"))
    # Add a softmax layer with 10 output units:
    model.add(layers.Dense(10, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.RMSprop(0.01),
        loss=keras.losses.categorical_crossentropy,
        metrics=[keras.metrics.categorical_accuracy])
    return model


import ray
import numpy as np

NUM_BATCHES = 100 
NUM_ITERS = 1000

ray.init()

def random_one_hot_labels(shape):
    n, n_class = shape
    classes = np.random.randint(0, n_class, n)
    labels = np.zeros((n, n_class))
    labels[np.arange(n), classes] = 1
    return labels


# Use GPU wth
# @ray.remote(num_gpus=1)
@ray.remote
class Worker(object):
    def __init__(self):
        self.model = create_keras_model()
        self.dataset = np.random.random((1000, 32))
        self.labels = random_one_hot_labels((1000, 10))

    def train(self):
        history = self.model.fit(self.dataset, self.labels, verbose=False)
        return history.history

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        # For simplicity this does not handle the optimizer state.
        self.model.set_weights(weights)

# 单个节点训练
networkActor = Worker.remote()
for iteration in range(NUM_ITERS):
    result_object_ref = networkActor.train.remote()
   

# 多节点训练
actor_list = [Worker.remote() for _ in range(NUM_BATCHES)]
for iteration in range(NUM_ITERS):
    object_refs= ray.get([actor.train.remote() for actor in actor_list])
    weights = ray.get([actor.get_weights.remote() for actor in actor_list])

    averaged_weights = [sum([weight[i] for weight in weights]) / len(weights) for i in range(len(weights[0]))]

    weight_id = ray.put(averaged_weights)
    [actor.set_weights.remote(weight_id) for actor in actor_list]
