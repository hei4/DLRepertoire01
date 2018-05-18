# -*- coding: utf-8 -*-
import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from net import CNN


def main():
    parser = argparse.ArgumentParser(description='pytorch example: MNIST')
    parser.add_argument('--batch', '-b', type=int, default=100,
                        help='Number of images in each mini-batch')
    parser.add_argument('--model', '-m', default='model.pth',
                        help='Network Model')
    args = parser.parse_args()

    batch_size = args.batch

    print('show training log')
    
    df = pd.read_csv('train.log')
    plt.plot(df['epoch'], df['train/accuracy'], label='train/acc.', marker = "o")
    plt.plot(df['epoch'], df['test/accuracy'], label='test/acc.', marker = "o")

    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.savefig('accuracy.png')
    plt.show()

    transform = transforms.Compose(
        [transforms.ToTensor(),    # transform to torch.Tensor
        transforms.Normalize(mean=(0.5,), std=(0.5,))])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                          download=True, transform=transform)

    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                         download=True, transform=transform)

    dataset = trainset + testset

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('device:', device)

    # Load & Predict Test
    param = torch.load('model.pth')
    net = CNN() #読み込む前にクラス宣言が必要
    net.to(device)  # for GPU
    net.load_state_dict(param)

    true_list = []
    pred_list = []
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            true_list.extend(labels.tolist())
            images, labels = images.to(device), labels.to(device)  # for GPU

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)        
            pred_list.extend(predicted.tolist())

    acc = accuracy_score(true_list, pred_list)
    print('Predict... all data acc.: {:.3f}'.format(acc))

    confmat = confusion_matrix(y_true=true_list, y_pred=pred_list)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(confmat, cmap=plt.cm.Purples, alpha=0.8)
    for i in range(confmat.shape[0]):
        for j in range(confmat.shape[1]):
            if confmat[i, j] > 0:
                ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')

    plt.xlabel('predicted label')
    plt.ylabel('true label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()


if __name__ == '__main__':
    main()
