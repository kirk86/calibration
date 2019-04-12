import os
import datetime
import numpy as np
import tensorflow as tf
# from tensorflow.python.framework import ops
from src.visualization import visualize as plot
# from tensorflow.examples.tutorials.mnist import input_data
from utils.data_loader import read_data_sets
from utils.rprop import RPropOptimizer


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class NeuralNetwork(object):
    def __init__(self):
        ############################################################
        # params config                                            #
        ############################################################
        from sklearn import datasets
        self.x = tf.placeholder(
            tf.float32,
            [None, 3072],
            name='inputs'
        )
        self.y = tf.placeholder(tf.float32, [None, 10], name='outputs')
        (X, Y), (teX, teY) = tf.keras.datasets.cifar10.load_data()
        # X, Y = datasets.make_circles(70000, factor=0.5, noise=0.05)
        X = X / 255.0
        teX = teX / 255.0
        valX, valY = X[:10000], Y[:10000]
        X, Y = X[10000:], Y[10000:]
        # self.teX, self.teY = X[:10000], Y[:10000]
        self.teX, self.teY = teX.reshape(-1, 3072), teY
        # X, Y = X[10000:], Y[10000:]
        valY = tf.keras.utils.to_categorical(valY, 10)
        self.teY = tf.keras.utils.to_categorical(self.teY, 10)
        Y = tf.keras.utils.to_categorical(Y, 10)
        self.X, self.Y = X.reshape(-1, 3072), Y
        self.valX, self.valY = valX.reshape(-1, 3072), valY
        self.batch_id = 0
        self.epochs = 300
        # self.objective = getattr(tf.nn,
        #                          'softmax_cross_entropy_with_logits_v2')
        # self.optimizer = RPropOptimizer
        # self.optimizer = tf.train.GradientDescentOptimizer
        self.objective = tf.losses.sparse_softmax_cross_entropy
        self.optimizer = tf.train.AdamOptimizer
        self.network_type = 'vanilla'
        self.batch_size = 1024
        self.num_layers = 3
        self.num_hidden = [512, 256, 128]
        self.learning_rate = 1.0e-3
        self.regularization = 1.0e-4
        self.temp_constant = 1.5
        ############################################################
        # logging setup                                            #
        ############################################################
        os.environ['CUDA_VISIBLE_DEVICES'] = '1'
        self.net = {
            'title': self.network_type,
            'train': {
                'acc': [],
                'loss': [],
                'error': [],
                'reg': []
            },
            'valid': {
                'acc': [],
                'loss': [],
                'error': [],
                'reg': []
            },
            'test': {
                'acc': [],
                'loss': [],
                'error': [],
                'reg': []
            },
            'calibration': {
                'before': {
                    'NLL': [],
                    'ECE': []
                },
                'after': {
                    'NLL': [],
                    'ECE': []
                }
            }
        }
        self.temperature = tf.Variable(tf.ones(shape=[1]) * self.temp_constant)

        # Next: optimize the temperature w.r.t. NLL
        self.temperature_optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.temperature, method='L-BFGS-B', options={'maxiter': 100})

        with tf.variable_scope(self.network_type, reuse=False):
            self.logits, self.probabilities, self.accuracy, \
                self.loss, self.error, self.l2, self.reg, \
                self.train_op = self.__call__(self.x, self.y)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def __call__(self, inputs, outputs):
        def choose_shape(W, layer):
            if layer == 0:
                return inputs.shape.as_list()[1:] + [self.num_hidden[layer]]
            elif layer == self.num_layers:
                return W[layer - 1].shape.as_list()[1:] + \
                    outputs.shape.as_list()[1:]
            else:
                return W[layer - 1].shape.as_list()[1:] + \
                    [self.num_hidden[layer]]
        W = []
        b = []
        modules = []
        for layer in range(self.num_layers + 1):
            activation = tf.identity if layer == self.num_layers \
                else tf.nn.elu
            with tf.name_scope(f"layer-{layer}"):
                with tf.name_scope(f"W{layer}"):
                    W.append(
                        tf.get_variable(
                            name=f"W{layer}",
                            shape=choose_shape(W, layer),
                            initializer=tf.glorot_uniform_initializer()
                        ))
                    variable_summaries(W[layer])
                with tf.name_scope(f"b{layer}"):
                    b.append(
                        tf.get_variable(
                            name=f"b{layer}",
                            initializer=tf.zeros(
                                outputs.shape.as_list()[1:]
                                if layer == self.num_layers
                                else [self.num_hidden[layer]])
                        ))
                    variable_summaries(b[layer])
                with tf.name_scope(f"W{layer}x_plus_b{layer}"):
                    modules.append(
                        activation(
                            tf.matmul(
                                inputs
                                if layer == 0 else modules[layer - 1],
                                W[layer]
                            ) + b[layer],
                            name='layer{}-{}'.format(
                                layer,
                                'softmax_probs'
                                if layer == self.num_layers else 'relu'
                            )
                        ))
                    tf.summary.histogram(
                        'probabilities'
                        if layer == self.num_layers else 'activations',
                        tf.argmax(modules[layer], axis=1, name='targets')
                        if layer == self.num_layers else modules[layer]
                    )

        if self.network_type == "vanilla":
            loss = self.objective(
                labels=tf.argmax(outputs, axis=1), logits=modules[-1])
            reg = tf.constant(0.0)

        if self.network_type == "l2":
            reg = tf.nn.l2_loss(W[-2]) * self.regularization
            loss = self.objective(
                labels=outputs, logits=modules[-1], name='softmax_v2') + reg

        if self.network_type == "scatter_weights":
            W_shape = W[-2].shape.as_list()  # 256x2
            scaling_factor = tf.constant(1.) / W_shape[0]
            C = tf.eye(W_shape[0]) - scaling_factor * tf.ones(
                shape=[W_shape[0], W_shape[0]])  # 256x256
            S = tf.matmul(tf.matmul(W[-2], C, transpose_a=True), W[-2])  # 2x2
            reg = tf.abs(tf.reduce_sum(S)) * self.regularization
            loss = self.objective(
                labels=outputs, logits=modules[-1], name='softmax_v2') + reg

        if self.network_type == "scatter_embedding":
            scaling_factor = tf.constant(1.) / self.batch_size
            C = tf.eye(self.batch_size) - scaling_factor * tf.ones(
                shape=[self.batch_size, self.batch_size])  # 64x64
            S = tf.matmul(
                tf.matmul(modules[-2], C, transpose_a=True), modules[-2]
            )  # 2x2
            reg = tf.abs(tf.reduce_sum(S)) * self.regularization
            loss = self.objective(
                labels=outputs, logits=modules[-1], name='softmax_v2') + reg

        with tf.name_scope("train"):
            train_op = self.optimizer(self.learning_rate).minimize(loss)
        with tf.name_scope("metrics"):
            with tf.name_scope("correctly_predicted_labels"):
                correct_predict_labels = tf.equal(
                    tf.argmax(outputs, axis=1),
                    tf.argmax(modules[-1], axis=1)
                )
            with tf.name_scope("accuracy"):
                accuracy = tf.reduce_mean(
                    tf.cast(correct_predict_labels, tf.float32),
                    name='accuracy')
            with tf.name_scope("error"):
                # error = tf.reduce_sum(
                #     tf.pow(
                #         tf.argmax(outputs, axis=1) -
                #         tf.argmax(tf.nn.softmax(modules[-1]), axis=1),
                #         2
                #     ) / (2 * self.batch_size)
                # )
                error = tf.losses.mean_squared_error(
                    tf.argmax(outputs, axis=1),
                    tf.argmax(tf.nn.softmax(modules[-1]), axis=1)
                )
            # loss = tf.reduce_mean(loss, name='final_loss')
        tf.summary.scalar('cross_entropy_objective', loss)
        tf.summary.scalar("accuracy", accuracy)
        tf.summary.scalar("error", error)
        if not self.network_type == "vanilla":
            tf.summary.scalar("regularization_value", reg)
        logits = modules[-1]
        probab = tf.nn.softmax(modules[-1])
        penlayer_output = modules[-2]
        import pprint
        print("=== Network Architecture ===")
        pprint.pprint(modules)
        print("=== Network Architecture ===\n")
        del W, b, modules
        return logits, probab, accuracy, loss, error, penlayer_output,\
            reg, train_op

    def train(self):
        epos = []
        batch_acc = []
        batch_loss = []
        batch_error = []
        batch_reg = []
        n_train_batch = len(self.X) // self.batch_size
        for epoch in range(self.epochs):
            for mbatch in range(n_train_batch):
                epos.append(epoch)

                X = self.X[
                    (mbatch * self.batch_size):
                    self.batch_size * (mbatch+1)
                ]
                Y = self.Y[
                    (mbatch * self.batch_size):
                    self.batch_size * (mbatch+1)
                ]
                # print("Training shapes: {}, {}".format(X.shape, Y.shape))
                train_data = np.c_[X.reshape(-1, np.prod(X.shape[1:])), Y]
                np.random.shuffle(train_data)

                logits, prob, acc, loss, error, l2, reg, \
                    _ = self.sess.run(
                        [
                            self.logits,
                            self.probabilities,
                            self.accuracy,
                            self.loss,
                            self.error,
                            self.l2,
                            self.reg,
                            self.train_op,
                        ],
                        feed_dict={
                            self.x:
                            train_data
                            [:, :np.prod(X.shape[1:])].reshape(X.shape),
                            self.y: train_data
                            [:, np.prod(X.shape[1:]):].reshape(Y.shape)
                        }
                    )
                batch_acc.append(acc)
                batch_loss.append(loss)
                batch_error.append(error)
                batch_reg.append(reg)

            self.net['train']['acc'].append(np.array(batch_acc).mean())
            self.net['train']['loss'].append(np.array(batch_loss).mean())
            self.net['train']['error'].append(np.array(batch_error).mean())
            self.net['train']['reg'].append(np.array(batch_reg).mean())

            print(
                "{}, Epoch: {}, Accuracy: {}, "
                "Loss: {}, Error: {}, Reg: {}".format(
                    self.network_type, epoch,
                    self.net['train']['acc'][epoch],
                    self.net['train']['loss'][epoch],
                    self.net['train']['error'][epoch],
                    self.net['train']['reg'][epoch])
            )

            ########################################################
            # validation                                           #
            ########################################################
            _ = self.predict(epoch, mode='valid')

            ########################################################
            # test                                                 #
            ########################################################
            # if epoch > 0 and epoch % 25 == 0:
            #     _ = self.predict(epoch // 25 - 1, mode='test')

        # return epos, self.net

    def predict(self, epoch=None, batch_xs=None, batch_ys=None,
                mode='valid', verbose=True):
        n_samples = len(self.valX) if mode == 'valid' \
            else len(self.teX) if mode == 'test' else len(batch_xs)

        logits = []
        labels = []
        batch_acc = []
        batch_loss = []
        batch_error = []
        batch_reg = []
        # this for loop is equivalent to one epoch, going through all data
        n_batches = n_samples // self.batch_size
        for mbatch in range(n_batches):
            if mode == 'valid':
                X = self.valX[
                    (mbatch*self.batch_size): self.batch_size * (mbatch+1)
                ]
                Y = self.valY[
                    (mbatch*self.batch_size): self.batch_size * (mbatch+1)
                ]
            elif mode == 'test':
                X = self.teX[
                    (mbatch*self.batch_size): self.batch_size * (mbatch+1)
                ]
                Y = self.teY[
                    (mbatch*self.batch_size): self.batch_size * (mbatch+1)
                ]
            else:
                X = batch_xs[
                    (mbatch*self.batch_size): self.batch_size * (mbatch+1)
                ]
                Y = batch_ys[
                    (mbatch*self.batch_size): self.batch_size * (mbatch+1)
                ]

            test_data = np.c_[X, Y]
            np.random.shuffle(test_data)
            logts, acc, loss, error, reg = self.sess.run(
                [
                    self.logits,
                    self.accuracy,
                    self.loss,
                    self.error,
                    self.reg,
                ],
                feed_dict={self.x: test_data[:, :3072],
                           self.y: test_data[:, 3072:]}
            )
            logits.append(logts)
            labels.append(Y)
            batch_acc.append(acc)
            batch_loss.append(loss)
            batch_error.append(error)
            batch_reg.append(reg)

        self.net[mode]['acc'].append(np.array(batch_acc).mean())
        self.net[mode]['loss'].append(np.array(batch_loss).mean())
        self.net[mode]['error'].append(np.array(batch_error).mean())
        self.net[mode]['reg'].append(np.array(batch_reg).mean())
        if verbose:
            print("{}, Epoch: {}, Accuracy: {}, "
                  "Loss: {}, Error: {}, Reg: {}".format(
                      mode, epoch, self.net[mode]['acc'][epoch],
                      self.net[mode]['loss'][epoch],
                      self.net[mode]['error'][epoch],
                      self.net[mode]['reg'][epoch]))

        logits = np.array(logits).reshape(len(logits) * logits[0].shape[0], -1)
        labels = np.array(labels).reshape(len(labels) * labels[0].shape[0], -1)
        return logits, labels

    def ece_loss(self, logits, labels, n_bins=15):
        """
        Calculates the Expected Calibration Error of a model.
        (This isn't necessary for temperature scaling, just a cool metric).
        The input to this loss is the logits of a model, NOT the softmax scores.
        This divides the confidence outputs into equally-sized interval bins.
        In each bin, we compute the confidence gap:
        bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
        We then return a weighted average of the gaps, based on the number
        of samples in each bin
        See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
        "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
        2015.
        n_bins (int): number of confidence interval bins
        """
        bin_boundaries = tf.linspace(0.0, 1.0, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        # softmaxes = tf.nn.softmax(logits, axis=1)
        softmaxes = tf.nn.softmax(logits)
        confidences = tf.reduce_max(softmaxes, axis=1)
        predictions = tf.argmax(softmaxes, axis=1)
        accuracies = tf.equal(
            predictions,
            tf.argmax(labels, axis=1)
            if len(labels.shape.as_list()) >= 2 else labels)

        ece = tf.zeros(shape=[1], dtype=tf.float32)
        try:
            if bin_lowers.shape.as_list()[0] != bin_uppers.shape.as_list()[0]:
                raise ValueError(
                    "Shape mismatch bin indices bin_lowers and bin_uppers")
            else:
                num_elems = bin_lowers.shape.as_list()[0]
        except ValueError as e:
            print(e)
            return -1

        # for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        for bin_idx in range(num_elems):
            in_bin = tf.logical_and(
                tf.greater(confidences, bin_lowers[bin_idx]),
                tf.less_equal(confidences, bin_uppers[bin_idx]))

            prop_in_bin = tf.reduce_mean(tf.cast(in_bin, tf.float32))

            mask_accuracies = tf.boolean_mask(accuracies, in_bin)
            accuracy_in_bin = tf.cond(
                tf.equal(tf.size(mask_accuracies),
                         0), lambda: tf.zeros(shape=[1], dtype=tf.float32),
                lambda: tf.reduce_mean(tf.cast(mask_accuracies, tf.float32)))

            mask_confidences = tf.boolean_mask(confidences, in_bin)
            avg_confidence_in_bin = tf.cond(
                tf.equal(tf.size(mask_confidences),
                         0), lambda: tf.zeros(shape=[1], dtype=tf.float32),
                lambda: tf.reduce_mean(tf.cast(mask_confidences, tf.float32)))
            ece += tf.abs(avg_confidence_in_bin -
                          accuracy_in_bin) * prop_in_bin

        self.sess.run(tf.global_variables_initializer())
        return self.sess.run(ece)[0]

    def temperature_scale(self, logits):
        """
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
        Perform temperature scaling on logits
        """
        return logits / self.temperature

    def set_temperature(self, logits, labels):
        """
        Tune the temperature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        logits = tf.Variable(logits) if isinstance(logits,
                                                   np.ndarray) else logits
        labels = tf.Variable(labels) if isinstance(labels,
                                                   np.ndarray) else labels
        labels = tf.argmax(labels, axis=1)

        # Calculate NLL and ECE before temperature scaling
        before_temperature_nll = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits))
        # output is scalar value and not tf symbolic variable
        before_temperature_ece = self.ece_loss(logits=logits, labels=labels)
        print('Before temperature - NLL: %.3f, ECE: %.3f' %
              (self.sess.run(before_temperature_nll), before_temperature_ece))
        self.net['calibration']['before']['NLL'].append(
            self.sess.run(before_temperature_nll))
        self.net['calibration']['before']['ECE'].append(before_temperature_ece)

        variables = []
        to_be_removed = [
            value
            if value.name.startswith('network') else variables.append(value)
            for value in tf.trainable_variables()
        ]

        del to_be_removed

        def evaluate(variables):
            loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.temperature_scale(logits), labels=labels))
            return loss

        self.temperature_optimizer.minimize(
            session=self.sess, step_callback=evaluate)

        print('Optimal temperature: %.3f' % self.sess.run(self.temperature))

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.temperature_scale(logits), labels=labels))
        # output is scalar and not tf symbolic varialbe
        after_temperature_ece = self.ece_loss(
            logits=self.temperature_scale(logits), labels=labels)
        print('After temperature - NLL: %.3f, ECE: %.3f' %
              (self.sess.run(after_temperature_nll), after_temperature_ece))
        self.net['calibration']['after']['NLL'].append(
            self.sess.run(after_temperature_nll))
        self.net['calibration']['after']['ECE'].append(after_temperature_ece)

        return self.temperature_scale(logits), labels


if __name__ == "__main__":
    nets = []
    nn = NeuralNetwork()
    nn.train()
    # epochs, network = nn.train()
    # nets.append(network)
