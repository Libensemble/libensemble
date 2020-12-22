import os
import numpy as np
import tensorflow as tf


__all__ = ['distrib_ml_eval_model']

def distrib_ml_eval_model(H, persis_info, sim_specs, libE_info):

    eval_steps = sim_specs['user']['eval_steps']
    model_file = H['model_file'][0][0]
    H_o = np.zeros(1, dtype=sim_specs['out'])

    _ , (mnist_test_images, mnist_test_labels) = \
        tf.keras.datasets.mnist.load_data(path='mnist.npz')

    dataset = tf.data.Dataset.from_tensor_slices((tf.cast(mnist_test_images[..., tf.newaxis] / 255.0, tf.float32), tf.cast(mnist_test_labels, tf.int64)))
    dataset = dataset.repeat().shuffle(10000).batch(64)

    model_path = os.path.abspath(model_file)

    mnist_model = tf.keras.models.load_model(model_path, compile=True)
    [loss, accuracy] = mnist_model.evaluate(dataset, steps=eval_steps)

    H_o['loss'] = loss
    H_o['accuracy'] = accuracy

    return H_o, persis_info
