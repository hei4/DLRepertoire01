# -*- code: utf-8 -*-
import argparse
import numpy as np
import cupy as cp
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sbn
from sklearn.metrics import confusion_matrix, accuracy_score

import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions

from net import MLP, CNN, CNNList


DEFAULT_LOAD = 'result/snapshot_iter_12000'


def main():
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default=DEFAULT_LOAD,
                        help='Resume the training from snapshot')
    parser.add_argument('--ch', '-c', type=int, default=256,
                        help='Number of units')
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# channels: {}'.format(args.ch))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('')

    # net = MLP(args.unit, 10)
    # net = CNN(args.ch, 10)
    net = CNNList(args.ch, 10)
    
    chainer.serializers.load_npz(args.resume, net, path='updater/model:main/predictor/')
    model = L.Classifier(net)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
        xp = cp
    else:
        xp = np
    predictor = model.predictor

    _, valid = chainer.datasets.get_mnist(ndim=3)    # for MLP, ~.get_mnist(ndim=1)

    true_list = []
    pred_list = []
    for i, (image, label) in enumerate(valid):
        newdim = [1]
        newdim.extend(image.shape)  # [1, 1, w, h]
        image = xp.array(image.reshape(newdim))

        if i % args.batchsize == 0:
            batch = image
        else:
            batch = xp.concatenate([batch, image], axis=0)
        
        if i % args.batchsize == args.batchsize - 1 or i == len(valid) - 1:
            with chainer.using_config('train', False), chainer.no_backprop_mode():
                pred = predictor(batch)
                pred = chainer.cuda.to_cpu(pred.data.argmax(axis=1))
                pred_list.extend(pred)
        
        true_list.append(label)

    # make&save DataFrame 
    df = pd.DataFrame({'true': true_list,
                       'pred': pred_list})
    print(df)
    df.to_csv('{}/pred.csv'.format(args.out), index=False)

    # calc acc.
    print('acc.', accuracy_score(true_list, pred_list))
    
    # make&save Confusion Matrix
    confmat = confusion_matrix(y_true=true_list, y_pred=pred_list)
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(confmat, cmap=plt.cm.Purples, alpha=0.8)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            if confmat[i, j] > 0:
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.tight_layout()
    
    plt.savefig('{}/confusion_matrix.png'.format(args.out), dpi=300)
    plt.show()
    

if __name__ == '__main__':
    main()

