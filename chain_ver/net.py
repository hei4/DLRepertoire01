# -*- code: utf-8 -*-
import chainer
import chainer.functions as F
import chainer.links as L


# Network definition
class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class CNN(chainer.Chain):
    def __init__(self, n_ch, n_out):
        super(CNN, self).__init__()
        with self.init_scope():
            he = chainer.initializers.HeNormal()
            self.cv1 = L.Convolution2D(None, n_ch // 8, ksize=3, stride=1, pad=1, initialW=he)
            self.cv2 = L.Convolution2D(None, n_ch // 4, ksize=3, stride=1, pad=1, initialW=he)
            self.fc1 = L.Linear(None, n_ch, initialW=he)
            self.fc2 = L.Linear(None, n_out, initialW=he)

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.cv1(x)), ksize=2)
        h = F.max_pooling_2d(F.relu(self.cv2(h)), ksize=2)
        h = F.relu(self.fc1(h))
        return self.fc2(h)


class ConvBlock(chainer.Chain):
    def __init__(self, n_out):
        super(ConvBlock, self).__init__()
        with self.init_scope():
            he = chainer.initializers.HeNormal()
            self.cv = L.Convolution2D(None, n_out, ksize=3, stride=1, pad=1, initialW=he)

    def __call__(self, x):
        return F.max_pooling_2d(F.relu(self.cv(x)), ksize=2)


class FullConBlock(chainer.Chain):
    def __init__(self, n_out, use_activation=True):
        super(FullConBlock, self).__init__()
        with self.init_scope():
            self.use_activation = use_activation
            he = chainer.initializers.HeNormal()
            self.fc = L.Linear(None, n_out, initialW=he)

    def __call__(self, x):
        h = self.fc(x)
        if self.use_activation:
            h = F.relu(h)
        return h


class CNNList(chainer.ChainList):
    def __init__(self, n_ch, n_out):
        super(CNNList, self).__init__()
        with self.init_scope():
            self.add_link(ConvBlock(n_ch // 8))
            self.add_link(ConvBlock(n_ch // 4))
            self.add_link(FullConBlock(n_ch))
            self.add_link(FullConBlock(n_ch, False))

    def __call__(self, x):
        for f in self:
            x = f(x)
        return x
