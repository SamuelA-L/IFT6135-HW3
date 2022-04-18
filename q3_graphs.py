import numpy as np
import matplotlib.pyplot as plt


def plot(title, arrays, y_label, xlabel='epoch', save=False, plot=True):

    fig, ax = plt.subplots()
    ax.plot(arrays[:, 0], label='Loss3 ', color='orange')
    ax.set_xlabel(xlabel)
    # ax.set_ylabel('loss', color='orange')
    ax2=ax.twinx()
    ax2.plot(arrays[:, 1], label='Accuracy ')
    ax2.set_ylabel('KNN accuracy', )
    ax.set_ylabel('loss2')
    # plt.title(title)


    if save:
        plt.savefig('/home/samuel/Desktop/figures_hw3/' + title + '.png')
    if plot:
        plt.show()


loss_vs_knn_acc_grad_stop = np.load('loss_acc_1.npy')
loss_vs_knn_acc_wo_grad_stop = np.load('nogradstop_loss_acc_1.npy')
imap_loss_vs_knn = np.load('gradstop_imap_loss_acc_1.npy')

plot("Loss vs KNN accuracy with gradient stopping",loss_vs_knn_acc_grad_stop, 'loss', save=True, plot=True)
plot("Loss vs KNN accuracy without gradient stopping",loss_vs_knn_acc_wo_grad_stop, 'loss', save=True, plot=True)
plot("Loss vs KNN accuracy with imap",imap_loss_vs_knn, 'loss', save=True, plot=True)