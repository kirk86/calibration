import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from utils.parameter import AppConfig
from sklearn.utils.extmath import softmax

config = AppConfig('./settings/config.yaml', 'appconfig')
fig_dir = os.path.join(config.work_dir, config.report_dir, 'figures')


def get_plot_buffer(data, labels):
    import io
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    buff = io.BytesIO()
    plt.savefig(buff, format='png')
    plt.close()
    buff.seek(0)
    return buff


def plot_metrics(nets, metric='train', title=None):
    import math
    rows = math.ceil(len(nets) / 2)
    fig, axes = plt.subplots(rows, 2, constrained_layout=True)
    for ax, net in zip(axes.ravel(), nets):
        save_dir = "{}/{}/{}_metrics".format(fig_dir, net.dataset, metric)
        ax.plot(net.net[metric]['acc'], '-r', alpha=0.7)
        ax.plot(net.net[metric]['loss'], '-b', alpha=0.7)
        ax.legend(labels=('accuracy', 'loss'), loc='upper right')
        # ax.set_title(f"{net.net['title']}")
        ax.set_title(net.net['title'])
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Accuracy / Loss")
        ax.grid(True)
        fig.suptitle(title)
    fig.savefig(save_dir)
    with open(save_dir + ".pickle", "wb") as fp:
        pickle.dump(fig, fp)
    for net in nets:
        net.run.add_artifact(save_dir + ".png")
    fig.clear()
    plt.close()


def plot_scatter(netobj, data, labels, probabs, valid_loss=None,
                 epoch=None, name='scatter_plot'):

    save_dir = os.path.join(fig_dir, netobj.dataset, netobj.network_type)
    if not tf.gfile.Exists(save_dir):
        os.makedirs(save_dir)
    fig = plt.figure()
    plt.autoscale(True)
    plt.scatter(
        data[:, 0],
        data[:, 1],
        c=labels,
        s=probabs * 100,
        alpha=0.3,
        cmap='viridis')
    if epoch is not None:
        plt.title("{}, epoch={}, val.loss={}, 1K rand samp.".format(
            netobj.network_type, epoch, round(valid_loss, 3)))
    else:
        plt.title("{}".format(netobj.network_type))
    plt.colorbar()
    plt.xlabel('First Dimenion of Data')
    plt.ylabel('Second Dimenion of Data')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(save_dir + f"/{name}")
    with open(save_dir + f"/{name}.pickle", "wb") as fp:
        pickle.dump(fig, fp)
    plt.close()
    netobj.run.add_artifact(save_dir + f"/{name}.png")


def reliability_diagram(netobj, logits, labels, n_bins=15, scaled=False):
    """
    outputs - a tensorflow tensor (size n x num_classes) with the
    outputs from the final linear layer - not the softmaxes
    labels - a tensorflow tensor (size n) with the labels
    """
    logits = netobj.sess.run(logits) if isinstance(logits, tf.tensor) \
        else logits
    labels = netobj.sess.run(labels) if isinstance(labels, tf.tensor) \
        else labels
    labels = labels.argmax(axis=1) if len(labels.shape) >= 2  \
        else labels
    softmaxes = softmax(logits)
    confidences, pred_labels = softmaxes.max(axis=1), softmaxes.argmax(
        axis=1)
    # boolean vector indicating correct predicted labels
    accuracies = np.equal(pred_labels, labels)  # correct predictions
    print("# of corr. pred. samples {}".format(accuracies.sum()))
    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    ax[0].hist(confidences)
    ax[0].set_ylabel("# of samples")
    ax[0].set_xlabel("confidence")
    ax[0].set_xlim(0, 1)

    # reliability diagram
    bins = np.linspace(0, 1, n_bins + 1)
    true_bins = np.linspace(0, 1, n_bins)
    bins[-1] = 1.0001
    width = bins[1] - bins[0]
    bin_lowers = bins[:-1]
    bin_uppers = bins[1:]
    ece = 0.0
    bin_correct_preds = []
    bin_conf_scores = []
    gap_error = []
    actual_bins = []
    prop_in_bins = []
    eces = []
    correct_samples_in_bin = []
    total_samples_in_bin = []
    # for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
    #     print("lower: ", bin_lower, " <-> ", "upper: ", bin_upper)
    bin_indices = [
        np.greater_equal(confidences, bin_lower) * np.less(
            confidences, bin_upper)
        for bin_lower, bin_upper in zip(bins[:-1], bins[1:])
    ]

    bin_correct_preds = [
        np.mean(accuracies[bin_index])
        if accuracies[bin_index].size != 0 else 1e-3
        for bin_index in bin_indices
    ]

    bin_conf_scores = [
        np.mean(confidences[bin_index])
        if confidences[bin_index].size != 0 else 1e-3
        for bin_index in bin_indices
    ]

    print("correct pred. across each bin {}\n"
          "confidence score for each bin {}".format(
              bin_correct_preds, bin_conf_scores))
    confs = ax[1].bar(bins[:-1], bin_correct_preds, width=width)
    gaps = ax[1].bar(
        bins[:-1],
        np.array(bin_conf_scores) - np.array(bin_correct_preds),
        bottom=bin_correct_preds,
        color=[1, 0.7, 0.7],
        alpha=0.5,
        width=width,
        hatch='//',
        edgecolor='r')
    ax[1].plot([0, 1], [0, 1], '--', color='gray')
    ax[1].legend(
        [confs, gaps], ['outputs', 'gap'], loc='best', fontsize='small')

    ece = netobj.ece_loss(
        logits=tf.variable(logits)
        if isinstance(logits, np.ndarray) else logits,
        labels=tf.variable(labels)
        if isinstance(labels, np.ndarray) else labels)

    textstr = '$\mathrm{ece}=%.2f$' % (ece)
    print("computed ece from reliability plot: {}".format(ece))
    # these are matplotlib.patch.patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # place a text box in upper left in axes coords
    ax[1].text(
        0.4,
        0.1,
        textstr,
        transform=ax[1].transaxes,
        fontsize=14,
        verticalalignment='top',
        bbox=props)

    # clean up
    ax[1].set_ylabel('accuracy')
    ax[1].set_xlabel('confidence')
    algoname = netobj.run.config['default_params']['params']['networks']
    dataset = netobj.run.config['default_params']['params']['datasets']
    plt.suptitle(algoname + '_calibrated' if scaled else algoname)

    fname = "{}/{}/{}/reliab_calib".format(
        fig_dir, dataset, algoname
    ) if scaled else "{}/{}/{}/reliab".format(
        fig_dir, dataset, algoname
        )
    # fig.savefig(fname)
    plt.savefig(fname)
    with open(fname + ".pickle", "wb") as fp:
        pickle.dump(fig, fp)
    fig.clear()
    plt.close()
    netobj.run.add_artifact(fname + ".png")


def reliability(logits, labels, n_bins=15):
    """
    outputs - a tensorflow tensor (size n x num_classes) with the
    outputs from the final linear layer - not the softmaxes
    labels - a tensorflow tensor (size n) with the labels
    """
    labels = np.array([0, 0, 0, 1, 1, 3, 5, 8, 9, 9, 4, 4, 6, 4, 5, 2, 2, 3, 3, 4])
    pred_labels = np.array([1, 1, 0, 3, 1, 4, 6, 9, 8, 9, 3, 4, 5, 3, 6, 2, 0, 0, 3, 1])
    probs = np.array([0.3, 0.2, 0.54, 0.32, 0.45, 0.65, 0.65, 0.67, 0.76, 0.34, 0.43, 0.53, 0.61, 0.33, 0.21, 0.12, 0.27, 0.38, 0.39, 0.45])
    # softmaxes = softmax(logits)
    # confidences, pred_labels = softmaxes.max(axis=1), softmaxes.argmax(
    #     axis=1)
    print("mean confidence over the probabilities: {}".format(
        np.mean(probs)))
    # boolean vector indicating correct predicted labels
    accuracies = np.equal(pred_labels, labels)
    print("accuracies {}".format(accuracies.mean()))
    fig, ax = plt.subplots(1, 2, constrained_layout=True)
    ax[0].hist(probs)
    ax[0].set_ylabel("# of samples")
    ax[0].set_xlabel("confidence")

    # reliability diagram
    bins = np.linspace(0, 1, n_bins + 1)
    bins[-1] = 1.0001
    width = bins[1] - bins[0]
    # for bin_lower, bin_upper in zip(bins[:-1], bins[1:]):
    #     print("lower: ", bin_lower, " <-> ", "upper: ", bin_upper)
    bin_indices = [
        np.greater_equal(probs, bin_lower) * np.less(
            probs, bin_upper)
        for bin_lower, bin_upper in zip(bins[:-1], bins[1:])
    ]

    bin_correct_preds = [
        np.mean(accuracies[bin_index])
        if accuracies[bin_index].size != 0 else 1e-3
        for bin_index in bin_indices
    ]

    bin_conf_scores = [
        np.mean(probs[bin_index])
        if probs[bin_index].size != 0 else 1e-3
        for bin_index in bin_indices
    ]

    print("correct pred. across each bin {}\n"
          "confidence score for each bin {}".format(
              bin_correct_preds, bin_conf_scores))
    confs = ax[1].bar(bins[:-1], bin_correct_preds, width=width)
    gaps = ax[1].bar(
        bins[:-1],
        np.array(bin_conf_scores) - np.array(bin_correct_preds),
        bottom=bin_correct_preds,
        color=[1, 0.7, 0.7],
        alpha=0.5,
        width=width,
        hatch='//',
        edgecolor='r')
    ax[1].plot([0, 1], [0, 1], '--', color='gray')
    ax[1].legend(
        [confs, gaps], ['outputs', 'gap'], loc='best', fontsize='small')

    # ece = netobj.ece_loss(
    #     logits=tf.variable(logits)
    #     if isinstance(logits, np.ndarray) else logits,
    #     labels=tf.variable(labels)
    #     if isinstance(labels, np.ndarray) else labels)

    # textstr = '$\mathrm{ece}=%.2f$' % (ece)
    # print("computed ece from reliability plot: {}".format(ece))
    # # these are matplotlib.patch.patch properties
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # # place a text box in upper left in axes coords
    # ax[1].text(
    #     0.4,
    #     0.1,
    #     textstr,
    #     transform=ax[1].transaxes,
    #     fontsize=14,
    #     verticalalignment='top',
    #     bbox=props)

    # clean up
    ax[1].set_ylabel('accuracy')
    ax[1].set_xlabel('confidence')
    plt.show()
    # algoname = netobj.run.config['default_params']['params']['networks']
    # dataset = netobj.run.config['default_params']['params']['datasets']
    # plt.suptitle(algoname + '_calibrated' if scaled else algoname)

    # fname = "{}/{}/{}/reliab_calib".format(
    #     fig_dir, dataset, algoname
    # ) if scaled else "{}/{}/{}/reliab".format(
    #     fig_dir, dataset, algoname
    #     )
    # # fig.savefig(fname)
    # plt.savefig(fname)
    # with open(fname + ".pickle", "wb") as fp:
    #     pickle.dump(fig, fp)
    # fig.clear()
    # plt.close()
    # netobj.run.add_artifact(fname + ".png")
