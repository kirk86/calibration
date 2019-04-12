#!/usr/bin/env python
# coding=utf-8

# import os
# import glob
import random
import sacred
# import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
# from utils.reader import InputData
from src.visualization import visualize as plot
from utils.parameter import AppConfig, ModelParams
from tensorflow.python.framework import ops
# from tensorflow.contrib.learn import ModeKeys
from src.models.train_model import NeuralNetwork, sacredIngredients

tf.logging.set_verbosity(tf.logging.INFO)

exper = sacred.Experiment('regularization_experiment',
                          ingredients=[sacredIngredients])

exper.captured_out_filter = \
    sacred.utils.apply_backspaces_and_linefeeds

exper.observers.append(
    sacred.observers.MongoObserver.create(
        url='localhost:27017',
        db_name='MY_DB'))

exper.observers.append(
    sacred.observers.TelegramObserver.from_config(
        './settings/bot.json'
    )
)

config = AppConfig('./settings/config.yaml', 'appconfig')
params = ModelParams('./settings/params.yaml', 'params')
nets = []


@exper.command
def metrics():
    ############################################################
    # Accuracy and Loss metrics                                #
    ############################################################
    for metric in ['train', 'valid', 'test']:
        plot.plot_metrics(nets=nets, metric=metric, title="Train Acc. / Loss")


@exper.main
def main(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    neural_net = NeuralNetwork()
    neural_net.train()
    nets.append(neural_net)
    ############################################################
    # Calibration                                              #
    ############################################################
    val_logits, val_labels = neural_net.predict(
        epoch=0, mode='valid', verbose=False
    )
    plot.reliability_diagram(
        neural_net, logits=val_logits, labels=val_labels
    )
    val_logits_calib, val_labels_calib = neural_net.set_temperature(
        logits=val_logits, labels=val_labels
    )
    plot.reliability_diagram(
        neural_net, logits=val_logits_calib,
        labels=val_labels_calib, scaled=True
    )
    neural_net.sess.close()
    tf.reset_default_graph()
    ops.reset_default_graph()
    plt.close('all')
    del neural_net.sess, neural_net
    return metrics()


for idx, dataset in enumerate(params.datasets):
    for net in params.networks:
        if net == "convnet":
            if dataset == "mnist" or dataset == "fashion_mnist":
                exper.run(config_updates={
                    "default_params.params.datasets": dataset,
                    "default_params.params.networks": net,
                    "default_params.params.inputs": [None, 28, 28, 1],
                    "default_params.params.outputs": 10
                }, options={'--name': f"{dataset}_{net}_{params.optimizer}"})
            if dataset == "cifar10" or dataset == "cifar100":
                exper.run(config_updates={
                    "default_params.params.datasets": dataset,
                    "default_params.params.networks": net,
                    "default_params.params.inputs": [None, 32, 32, 3],
                    "default_params.params.outputs": params.outputs[idx],
                }, options={'--name': f"{dataset}_{net}_{params.optimizer}"})
        else:
            exper.run(config_updates={
                "default_params.params.datasets": dataset,
                "default_params.params.networks": net,
                "default_params.params.inputs": params.inputs[idx],
                "default_params.params.outputs": params.outputs[idx]
            }, options={'--name': f"{dataset}_{net}_{params.optimizer}"})

                    # "default_params.params.learning_rate": 1.0e-3,
                    # "default_params.params.optimizer": 'AdamOptimizer'
# exper.run_command('metrics')
                # "default_params.params.learning_rate": 1.0e-2,
                # "default_params.params.optimizer": 'AdamOptimizer'},


    # input_data = InputData(config, params)
    # model = tf.estimator.Estimator(model_fn=nade.model_fn, params=params)
        # while True:
        #     model.train(input_fn=lambda: input_data.input_fn(ModeKeys.TRAIN), steps=1000)
        #     results_gen = model.predict(input_fn=lambda: input_data.input_fn(ModeKeys.INFER))
        #     config.logger.info(input_data.decode(list(itertools.islice(results_gen, params.infer_batch_size))))
        #     # train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_data.input_fn(ModeKeys.TRAIN))
        #     # eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_data.input_fn(ModeKeys.EVAL))
        #     # tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
