import tensorflow as tf


def run(dataset):
    x = tf.placeholder(tf.float32, [None, 784])

    # Weights
    w = tf.Variable(tf.zeros([784, 10]))
    # Biases
    b = tf.Variable(tf.zeros([10]))

    # The model
    y = tf.nn.softmax(tf.matmul(x, w) + b)
    # Placeholder to input the correct answers
    y_ = tf.placeholder(tf.float32, [None, 10])

    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    init = tf.initialize_all_variables()

    sess = tf.Session()
    sess.run(init)

    for i in range(1000):
        batch_xs, batch_ys = dataset.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("Simple network accuracy", sess.run(accuracy, feed_dict={x: dataset.test.images, y_: dataset.test.labels}))
