import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# PS and worker hosts (Only if we create the worker server here)
# ps_hosts = ['localhost:2222']
# worker_hosts = ['localhost:2223']

# Create a cluster from the parameter server and worker hosts. (Only if we create the worker server here)
# cluster = tf.train.ClusterSpec({"ps": ps_hosts, "worker": worker_hosts})

# Model params
with tf.device("/job:ps/task:0"):
    b = tf.Variable(tf.truncated_normal([10]), name="b")
    W = tf.Variable(tf.truncated_normal([784, 10])*(1/tf.sqrt(784.0)), name="W")

with tf.device("/job:worker/task:0"):
    # Placement
    x = tf.placeholder(tf.float32, [None, 784])
    y = tf.placeholder(tf.float32, [None, 10])

    # Model (toy example)
    relu = tf.nn.relu(tf.matmul(x, W) + b)
    ce_cost = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=relu)
    cross_entropy = tf.reduce_mean(ce_cost)

    # Optimizer
    learning_rate = 0.001
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_step = optimizer.minimize(cross_entropy)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(relu, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Create local server and attach session to this server (Only if we create the worker server here)
# server = tf.train.Server(cluster, job_name="worker", task_index=0)
# sess = tf.Session(server.target)

# Connect to server and attach session
sess = tf.Session("grpc://localhost:2223")

# Run training n iterations
sess.run(tf.global_variables_initializer())
for step in range(1000) :
    batch_xs, batch_ys = mnist.train.next_batch(100)
    cross_entropy_, _ = sess.run([cross_entropy, train_step], feed_dict={x: batch_xs, y: batch_ys})
    test_acc_ = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print step, cross_entropy_, test_acc_
