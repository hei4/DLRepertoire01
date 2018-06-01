# -*- coding: utf-8 -*-
import numpy as np
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# ジェネレーターを返す関数
def batch_generator(X, y, batch_size=100, shuffle=False, random_seed=None):
    idx = np.arange(y.shape[0])

    if shuffle:
        rng = np.random.RandomState(random_seed)
        rng.shuffle(idx)
        X = X[idx]
        y = y[idx]

    for i in range(0, X.shape[0], batch_size):
        yield (X[i:i+batch_size, :], y[i:i+batch_size, :])


class ConvNN(object):
    def __init__(self, batchsize=100, epochs=20, learning_rate=1e-4,
                 dropout_rate=0.5, shuffle=True, random_seed=None,
                 n_features=None, n_classes=None):
        np.random.seed(0)
        self.batchsize = batchsize
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.shuffle = shuffle
        
        self.n_features = n_features
        self.n_classes = n_classes

        g = tf.Graph()
        with g.as_default():
            # 乱数シードを設定
            tf.set_random_seed(random_seed)
            # モデルを構築
            self.build()
            # 変数を初期化
            self.init_op = tf.global_variables_initializer()
            # saver
            self.saver = tf.train.Saver()

        # セッションを作成
        self.sess = tf.Session(graph=g)

    def build(self):
        # Xとyのプレースホルダーを作成
        tf_x = tf.placeholder(tf.float32, shape=[None, self.n_features], name='tf_x')
        tf_y = tf.placeholder(tf.int32, shape=None, name='tf_y')
        is_train = tf.placeholder(tf.bool, shape=(), name='is_train')

        # xを4次元テンソルに変換：[バッチサイズ, 幅, 高さ, チャネル=1]
        tf_x_image = tf.reshape(tf_x, shape=[-1, 28, 28, 1],
                                name='input_x_2dimages')

        # 第1層：畳み込み層１
        h1 = tf.layers.conv2d(tf_x_image, kernel_size=(3, 3), filters=32, activation=tf.nn.relu,
                              kernel_initializer=tf.keras.initializers.he_normal())

        # 最大値プーリング
        h1_pool = tf.layers.max_pooling2d(h1, pool_size=(2, 2), strides=(2, 2))

        # 第2層：畳み込み層2
        h2 = tf.layers.conv2d(h1_pool, kernel_size=(3, 3), filters=64, activation=tf.nn.relu,
                              kernel_initializer=tf.keras.initializers.he_normal())
        
        # 最大値プーリング
        h2_pool = tf.layers.max_pooling2d(h2, pool_size=(2, 2), strides=(2, 2))
        tf.keras.initializers.he_normal
        # 第3層：全結合層1
        input_shape = h2_pool.get_shape().as_list()
        n_input_units = np.prod(input_shape[1:])
        h2_pool_flat = tf.reshape(h2_pool, shape=[-1, n_input_units])
        h3 = tf.layers.dense(h2_pool_flat, 256, activation=tf.nn.relu,
                             kernel_initializer=tf.keras.initializers.he_normal())
        
        # 第4層：全結合層2（線形活性化）
        h4 = tf.layers.dense(h3, self.n_classes, activation=None,
                             kernel_initializer=tf.keras.initializers.he_normal())   # h4 = logits
        
        # 予測
        predictions = {
            'probabilities': tf.nn.softmax(h4, name='probabilities'),
            'labels': tf.argmax(h4, axis=1, name='labels')
        }
                
        # 損失関数と最適化
        cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=h4, labels=tf_y),
            name='cross_entropy_loss')
        
        # オプティマイザ
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(cross_entropy_loss, name='train_op')
        
        '''
        # 損失関数        
        cross_entropy_loss = tf.reduce_mean(
            tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=h4), name='cross_entropy_loss')
                                                             
        # オプティマイザ
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = optimizer.minimize(loss=cross_entropy_loss, name='train_op')
        '''
        
        # 予測正解率を特定
        correct_predictions = tf.equal(predictions['labels'], tf.argmax(tf_y, axis=1), name='correct_preds')
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name='accuracy')
        
    def save(self, epoch, path='./tflayers-model/'):
        if not os.path.isdir(path):
            os.makedirs(path)
        
        print('Saving model in {}'.format(path))
        self.saver.save(self.sess, os.path.join(path, 'model.ckpt'), global_step=epoch)
        
    def load(self, epoch, path):
        print('Loading model from {}'.format(path))
        self.saver.restore(self.sess, os.path.join(path, 'model.ckpt-{}'.format(epoch)))
    
    def train(self, training_set, validation_set=None, initialize=True):
        # 変数を初期化
        if initialize:
            self.sess.run(self.init_op)
        
        self.train_cost = []
        X_data = np.array(training_set[0])
        y_data = np.array(training_set[1])
        
        for epoch in range(1, self.epochs + 1):
            batch_gen = batch_generator(X_data, y_data, shuffle=self.shuffle)
            avg_loss = 0.0
            for i, (batch_x, batch_y) in enumerate(batch_gen):
                feed = {'tf_x:0': batch_x,
                        'tf_y:0': batch_y,
                        'is_train:0': True}     # ドロップアウト
                
                loss, _ = self.sess.run(['cross_entropy_loss:0', 'train_op'], feed_dict=feed)
                avg_loss += loss
            
            print('Epoch {}: Training Avg. Loss: {}'.format(epoch, avg_loss))
        
            if validation_set is not None:
                feed = {'tf_x:0': batch_x,
                        'tf_y:0': batch_y,
                        'is_train:0': False}    # ドロップアウト
                valid_acc = self.sess.run('accuracy:0', feed_dict=feed)
                print('Validation Acc: {}'.format(valid_acc))
            
            print()
            
    def predict(self, X_test, return_prob=False):
        feed = {'tf_x:0': X_test,
                'is_train:0': False}    # ドロップアウト
        
        if return_prob:
            return self.sess.run('probabilities:0', feed_dict=feed)
        else:
            return self.sess.run('labels:0', feed_dict=feed)


def main():
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    X_train = mnist.train.images[:50000, :]
    y_train = mnist.train.labels[:50000, :]

    X_test = mnist.test.images
    y_test = mnist.test.labels

    X_valid = mnist.train.images[50000:, :]
    y_valid = mnist.train.labels[50000:, :]

    print('train data:',
          type(X_train), X_train.shape, type(X_train[0]),
          type(y_train), y_train.shape, type(y_train[0]))
    print('test data:',
          type(X_test), X_test.shape, type(X_test[0]),
          type(y_test), y_test.shape, type(y_test[0]))
    print('valid data:',
          type(X_valid), X_valid.shape, type(X_valid[0]),
          type(y_valid), y_valid.shape, type(y_valid[0]))
    print()

    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]

    cnn = ConvNN(random_seed=0, n_features=n_features, n_classes=n_classes)

    # モデルのトレーニング
    cnn.train(training_set=(X_train, y_train),
              validation_set=(X_valid, y_valid))
    cnn.save(epoch=20)

    del cnn

    cnn2 = ConvNN(random_seed=0, n_features=n_features, n_classes=n_classes)
    cnn2.load(epoch=20, path='./tflayers-model/')
    print(cnn2.predict(X_test[:10, :]))

    y_pred = cnn2.predict(X_test)

    print('Test Accuracy: {}'.format(
        np.sum(y_pred == np.argmax(y_test, axis=1)) / len(y_test)
    ))


if __name__ == '__main__':
    main()

