import numpy as nump
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):

    # 28x28 pixel images with only one channel, batch size dynamically calculated (-1) based on inputs in x
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Apply 32 5x5 filters to the input layer, ReLU activation is applied
    conv_layer_one = tf.layers.conv2d(
        inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    # Attach layer that performs max pooling using a 2x2 filter
    pool_layer_one = tf.layers.max_pooling2d(inputs=conv_layer_one, pool_size=[2, 2], strides=2)

    # 64 5x5 filters with ReLU activation and second pooling layer
    conv_layer_two = tf.layers.conv2d(
        inputs=pool_layer_one, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool_layer_two = tf.layers.max_pooling2d(inputs=conv_layer_two, pool_size=[2, 2], strides=2)

    # Flatten tensor to only 2 dimensions, pool_layer_two width * pool_layer_two height * 64 channels
    pool_layer_two_dense = tf.reshape(pool_layer_two, [-1, 7 * 7 * 64])

    # Dense layer with 1024 neurons
    dense = tf.layers.dense(inputs=pool_layer_two_dense, units=1024, activation=tf.nn.relu)

    #Dropout only if training is true
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # For digits we only need 10 (0-9) final layers
    logits = tf.layers.dense(inputs=dropout, units=10)

    # tf.argmax is the element in the corresponding row of the logits tensor with the highest raw value
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images
    train_labels = nump.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images
    eval_labels = nump.asarray(mnist.test.labels, dtype=np.int32)
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/temp/mnist_convnet_model")

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data}, y=train_labels, batch_size=100, num_epochs=None, shuffle=True)

    mnist_classifier.train(input_fn=train_input_fn, steps=20000, hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()