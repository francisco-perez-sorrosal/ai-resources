import sys
import argparse
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Network config
n_input = 784
n_hidden_1 = 64
n_classes = 10

# Experiment config
batch_size = 100
training_epochs = 20
reporting_freq = 100

# Cluster specification: PS and worker hosts
ps_hosts = ['localhost:2222', 'localhost:2223']
worker_hosts = ['localhost:2232']
cluster = tf.train.ClusterSpec({"ps":ps_hosts, "worker":worker_hosts})

FLAGS = None

def main(_):
    print("Flags %s" % FLAGS)

    # Start a server for a specific task
    server = tf.train.Server(cluster,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == "ps":
        print("Starting param server name: {}".format(FLAGS.task_index))
        print("Server definition: {}".format(server.server_def))
        server.join()
        exit(0)

    print("Starting worker server name: {}".format(FLAGS.task_index))

    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # Model params
    with tf.device("/job:ps/task:0"):
        W = tf.Variable(tf.truncated_normal([n_input, n_hidden_1]), name="W")
        b = tf.Variable(tf.truncated_normal([1, n_hidden_1]), name="b")

    with tf.device("/job:ps/task:1"):
        W2 = tf.Variable(tf.truncated_normal([n_hidden_1, n_classes]), name="W2")
        b2 = tf.Variable(tf.truncated_normal([1, n_classes]), name="b2")

    with tf.device("/job:worker/task:%d" % FLAGS.task_index):
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
        global_step = tf.get_variable('global_step', [],
                                    initializer = tf.constant_initializer(0),
                                    trainable = False)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy, global_step=global_step)

        # Accuracy
        correct_prediction = tf.equal(tf.argmax(relu, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        init_op = tf.global_variables_initializer()
        print("Variables initialized ... ")

    batch_count = int(mnist.train.num_examples / batch_size)
    print("Training epochs: %d" % training_epochs)
    print("Batch count per training epoch: %d" % batch_count)

    sv = tf.train.Supervisor(is_chief=(FLAGS.task_index == 0),
                            global_step=global_step,
                            init_op=init_op)

    # Connect to server and attach session
    begin_exp_time = time.time()
    with sv.prepare_or_wait_for_session(server.target) as sess:

        # Run training n iterations
        start_time = time.time()
        for epoch in range(training_epochs) :
            count = 0
            for i in range(batch_count):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                cross_entropy_, _ , step = sess.run([cross_entropy, train_step, global_step], feed_dict={x: batch_xs, y: batch_ys})
                test_acc_ = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
                count += 1
                if count % reporting_freq == 0 or i + 1 == batch_count:
                    elapsed_time = time.time() - start_time
                    start_time = time.time()
                    print("Step: %d," % (step + 1),
                          " Epoch: %2d of %2d," % (epoch + 1, training_epochs),
                          " Batch: %3d of %3d," % (i + 1, batch_count),
                          " Cost: %.4f," % cross_entropy_,
                          " AvgTime: %3.2fms" % float(elapsed_time * 1000 / reporting_freq))
                    count = 0

        print("Test-Accuracy: %2.2f" % test_acc_)
        print("Total Time: %3.2fs" % float(time.time() - begin_exp_time))
        print("Final Cost: %.4f" % cross_entropy_)
    sv.stop()
    print("Experiment finished!!!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--job_name",
        type=str,
        default="",
        help="Either 'ps' or 'worker'"
    )
    parser.add_argument(
        "--task_index",
        type=int,
        default=0,
        help="Index of task within the job"
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)