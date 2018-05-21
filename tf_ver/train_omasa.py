import numpy as np
import tensorflow as tf

#global variants
BASE_CHANNEL = 32
HIDDEN_UNITS = 256
CLASS_NUM = 10
EPOCH = 500
BATCH_SIZE = 100


#load data from npz file
npz_xd = np.load("mnist_part_xd.npz")
print('type(npz_xd)' + str(type(npz_xd)))
print('type(npz_xd["x"])=' + str(type(npz_xd["x"])))
print('type(npz_xd["d"])=' + str(type(npz_xd["d"])))
print('npz_xd["x"].shape=' + str(npz_xd["x"].shape))
print('npz_xd["d"].shape=' + str(npz_xd["d"].shape))
print('npz_xd["x"][0].shape=' + str(npz_xd["x"][0].shape))
print('npz_xd["x"][0][0]=' + str(npz_xd["x"][0][0]))
print('npz_xd["d"][0]=' + str(npz_xd["d"][0]))

#make input data, target data
x = npz_xd["x"]
d = npz_xd["d"]

#change target data from [N] to [N, 10]
def make_target(d):
    d_mod = np.zeros((d.shape[0], 10), dtype=np.float32)
    for num, data in enumerate(d):
        d_mod[num][int(data)] = 1.0
    print("d_mod.shape = " + str(d_mod.shape))
    print("d_mod = " + str(d_mod))
    return d_mod

d_mod = make_target(d)

x_train = x[0:5000]
x_test = x[5000:6000]
d_train = d_mod[0:5000]
d_test = d_mod[5000:6000]

#computation graph
x_ = tf.placeholder(tf.float32, [None, 784])
d_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x_, [-1, 28, 28, 1])#reshape for convolution


with tf.name_scope("conv1"):
    w1 = tf.Variable(tf.truncated_normal([3, 3, 1, BASE_CHANNEL], mean=0.0, stddev=0.05), dtype=tf.float32)
    b1 = tf.Variable(tf.zeros([BASE_CHANNEL]), dtype=tf.float32)
    conv1 = tf.nn.conv2d(x_image, w1, strides=[1, 1, 1, 1], padding="SAME") + b1
    relu1 = tf.nn.relu(conv1)

with tf.name_scope("pool1"):
    pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("conv2"):
    w2 = tf.Variable(tf.truncated_normal([3, 3, BASE_CHANNEL, BASE_CHANNEL * 2], mean=0.0, stddev=0.05), dtype=tf.float32)
    b2 = tf.Variable(tf.zeros([BASE_CHANNEL * 2]), dtype=tf.float32)
    conv2 = tf.nn.conv2d(pool1, w2, strides=[1, 1, 1, 1], padding="SAME") + b2
    relu2 = tf.nn.relu(conv2)

with tf.name_scope("pool2"):
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.name_scope("fc3"):
    # pool2_node = tf.shape(pool2)[1] * tf.shape(pool2)[2] * tf.shape(pool2)[3]
    w3 = tf.Variable(tf.random_normal([7 * 7 * BASE_CHANNEL * 2, HIDDEN_UNITS], mean=0.0, stddev=0.05), dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([HIDDEN_UNITS]), dtype=tf.float32)
    reshape3 = tf.reshape(pool2, [-1, 7 * 7 * BASE_CHANNEL * 2])
    u3 = tf.matmul(reshape3, w3) + b3
    relu3 = tf.nn.relu(u3)

with tf.name_scope("fc4"):
    w4 = tf.Variable(tf.random_normal([HIDDEN_UNITS, CLASS_NUM], mean=0.0, stddev=0.05), dtype=tf.float32)
    b4 = tf.Variable(tf.zeros([CLASS_NUM]), dtype=tf.float32)
    u4 = tf.matmul(relu3, w4) + b4
    prob = tf.nn.softmax(u4)

with tf.name_scope("loss"):
    loss = - tf.reduce_mean(tf.multiply(d_, tf.log(tf.clip_by_value(prob, 1e-10, 1e+30))), name='loss')

# train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


with tf.name_scope("accuracy"):
    correct_prediction = tf.equal(tf.argmax(prob, 1), tf.argmax(d_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('data10', sess.graph)

#make minibatch
def make_minibatch(per_list, x_data, d_data):
    x_data_mini = x_data[per_list]
    d_data_mini = d_data[per_list]
    return x_data_mini, d_data_mini

for epoch in range(EPOCH):
    
    x_random_list = np.random.permutation(len(x_train))
    for batch_count in range(0, len(x_train), BATCH_SIZE):
        
        batch_1_list = x_random_list[batch_count * BATCH_SIZE : (batch_count + 1) * BATCH_SIZE]
        x_minibatch, d_minibatch = make_minibatch(batch_1_list, x_train, d_train)

        sess.run(train_step, feed_dict={x_: x_minibatch, d_: d_minibatch})
    
    
    loss_, accu_ = sess.run([loss, accuracy], feed_dict={x_: x_train, d_: d_train})
    print('epoch =' + str(epoch) + ' ,training loss =' + str(loss_), ' ,training accuracy =', str(accu_))

    #test phase
    if epoch % 10 == 0:
        loss_test_, accu_test_= sess.run([loss, accuracy], feed_dict={x_: x_test, d_: d_test})
        print('test loss = ', str(loss_test_), ' ,test accuracy', str(accu_test_))

