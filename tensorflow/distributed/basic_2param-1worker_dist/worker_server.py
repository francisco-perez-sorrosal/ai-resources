import sys
import tensorflow as tf

ps_hosts = ['localhost:2222', 'localhost:2223']
work_hosts = ['localhost:2232']

task_number=0

cluster = tf.train.ClusterSpec({"ps" : ps_hosts, "worker" : work_hosts})
server = tf.train.Server(cluster, job_name="worker", task_index=task_number)

print("Starting worker server name: {}".format(task_number))
print("Server definition: {}".format(server.server_def))
server.join()
