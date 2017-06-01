import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

n_input = 784
n_hidden_1 = 64
n_classes = 10

# Model params
with tf.device("/job:ps/task:0"):
    W = tf.Variable(tf.truncated_normal([n_input, n_hidden_1]), name="W")
    b = tf.Variable(tf.truncated_normal([1, n_hidden_1]), name="b")

with tf.device("/job:ps/task:1"):
    W2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_classes]), name="W2")
    b2 = tf.Variable(tf.truncated_normal([1, n_classes]), name="b2")

with tf.device("/job:worker/task:0"):
    # Placement
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # Model (toy example)
    layer_1 = tf.nn.relu(tf.matmul(x, W) + b)

    relu = tf.nn.relu(tf.matmul(layer_1, W2) + b2)
    ce_cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=relu)
    cross_entropy = tf.reduce_mean(ce_cost)

    # Optimizer
    learning_rate = 0.001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(relu, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Connect to server and attach session
sess = tf.Session("grpc://localhost:2232")

# Run training n iterations
sess.run(tf.global_variables_initializer())
for step in range(1000) :
    batch_xs, batch_ys = mnist.train.next_batch(100)
    cross_entropy_, _ = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y: batch_ys})
    test_acc_ = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print step, cross_entropy_, test_acc_