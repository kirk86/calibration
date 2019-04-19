import os
import sacred
import datetime
import numpy as np
import tensorflow as tf
# from tensorflow.python.framework import ops
from tensorflow.python.client import device_lib
from src.visualization import visualize as plot
# from tensorflow.examples.tutorials.mnist import input_data
# from utils.data_loader import read_data_sets
from utils.loader import data_loader
from utils.rprop import RPropOptimizer

sacredIngredients = sacred.Ingredient('default_params')

sacredIngredients.add_config('./settings/config.yaml')
sacredIngredients.add_config('./settings/params.yaml')

# tfe = tf.enable_eager_execution()


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


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
    @sacredIngredients.capture
    def __init__(self, appconfig, params, _run):
        self.run = _run
        ############################################################
        # app config                                               #
        ############################################################
        self.work_dir = appconfig['work_dir']
        self.data_dir = appconfig['data_dir']
        self.model_dir = appconfig['model_dir']
        self.log_dir = appconfig['log_dir']
        ############################################################
        # params config                                            #
        ############################################################
        self.x = tf.placeholder(
            tf.float32,
            params['inputs'] if isinstance(params['inputs'], list)
            else [None, params['inputs']],
            name='inputs'
        )
        self.y = tf.placeholder(tf.float32, [None, params['outputs']],
                                name='outputs')
        self.dataset = params['datasets']
        self.network_type = params['networks']
        self.data = data_loader(
            self.dataset,
            params['outputs'],
            reshape=True if self.network_type != "convnet" else False
        )
        self.batch_id = 0
        self.epochs = params['num_epochs']
        self.objective = getattr(tf.nn, params['objective'])
        self.optimizer = getattr(tf.train, params['optimizer'])
        # self.optimizer = RPropOptimizer
        self.batch_size = params['batch_size']
        self.num_layers = params['num_layers']
        self.num_hidden = params['num_hidden']
        self.learning_rate = params['learning_rate']
        self.regularization = params['regularization']
        self.temp_constant = params['temperature_constant']
        ############################################################
        # logging setup                                            #
        ############################################################
        time_string = datetime.datetime.now().isoformat()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(params['visible_devices'])
        self.logs = os.path.join(self.work_dir, self.log_dir)
        self.models = os.path.join(self.work_dir, self.model_dir)
        experiment_name = f"{self.dataset}/{self.network_type}/{self.epochs}_epochs_{time_string}"
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
        self.temperature = tf.Variable(
            tf.ones(shape=[1]) * self.temp_constant
        )
        # Next: optimize the temperature w.r.t. NLL
        self.temperature_optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.temperature, method='L-BFGS-B', options={'maxiter': 100})

        # with tf.Graph().as_default():
        with tf.variable_scope(self.network_type, reuse=False):
            if self.network_type == 'convnet':
                from src.models.convnet import lenet
                kernel_size = params['kernel_size']
                pool_size = params['pool_size']
                self.logits, self.probabilities, self.accuracy, \
                    self.loss, self.error, self.l2, self.reg, \
                    self.train_op = lenet(
                        self, self.x, self.y, kernel_size, pool_size)
            else:
                self.logits, self.probabilities, self.accuracy, \
                    self.loss, self.error, self.l2, self.reg, \
                    self.train_op = self.__call__(self.x, self.y)

        # self.saver = tf.train.Saver()
        # self.summary = tf.summary.merge_all()
        self.sess = tf.Session()
        self.train_writer = tf.summary.FileWriter(
            f"{self.logs}/train/{experiment_name}", self.sess.graph)
        self.valid_writer = tf.summary.FileWriter(
            f"{self.logs}/valid/{experiment_name}", self.sess.graph)
        self.test_writer = tf.summary.FileWriter(
            f"{self.logs}/test/{experiment_name}", self.sess.graph)
        self.sess.run(tf.global_variables_initializer())

    def __call__(self, inputs, labels):
        def choose_shape(W, layer):
            if layer == 0:
                return inputs.shape.as_list()[1:] + [self.num_hidden[layer]]
            elif layer == self.num_layers:
                return W[layer - 1].shape.as_list()[1:] + \
                    labels.shape.as_list()[1:]
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
                                labels.shape.as_list()[1:]
                                if layer == self.num_layers
                                else [self.num_hidden[layer]])
                        ))
                    variable_summaries(b[layer])
                with tf.name_scope(f"W{layer}x_plus_b{layer}"):
                    modules.append(
                        activation(
                            tf.matmul(
                                inputs if layer == 0
                                else modules[layer - 1], W[layer]
                            ) + b[layer],
                            name='layer{}-{}'.format(
                                layer,
                                'softmax_probs' if layer == self.num_layers
                                else 'relu'
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
                labels=labels, logits=modules[-1], name='softmax_v2'
            )
            reg = tf.constant(0.0)

        if self.network_type == "l2":
            reg = tf.nn.l2_loss(W[-2]) * self.regularization
            loss = self.objective(
                labels=labels, logits=modules[-1], name='softmax_v2'
            ) + reg

        if self.network_type == "scatter_weights":
            W_shape = W[-2].shape.as_list()  # 256x2
            scaling_factor = tf.constant(1.) / W_shape[0]
            C = tf.eye(W_shape[0]) - scaling_factor * tf.ones(
                shape=[W_shape[0], W_shape[0]])  # 256x256
            S = tf.matmul(tf.matmul(W[-2], C, transpose_a=True), W[-2])  # 2x2
            reg = tf.abs(tf.reduce_sum(S)) * self.regularization
            loss = self.objective(
                labels=labels, logits=modules[-1], name='softmax_v2'
            ) + reg

        if self.network_type == "scatter_embedding":
            # l2_shape = l2.shape.as_list()  # 64x2
            scaling_factor = tf.constant(1.) / self.batch_size
            C = tf.eye(self.batch_size) - (scaling_factor * tf.ones(
                shape=[self.batch_size, self.batch_size]
            ))  # 64x64
            S = tf.matmul(tf.matmul(modules[-2], C, transpose_a=True),
                          modules[-2])  # 2x2
            reg = tf.abs(tf.reduce_sum(S)) * self.regularization
            loss = self.objective(
                labels=labels, logits=modules[-1], name='softmax_v2'
            ) + reg
            loss = self.objective(
                labels=labels, logits=modules[-1], name='softmax_v2'
            )

        with tf.name_scope("train"):
            train_op = self.optimizer(self.learning_rate).minimize(loss)
        with tf.name_scope("metrics"):
            with tf.name_scope("correctly_predicted_labels"):
                correct_predict_labels = tf.equal(
                    tf.argmax(labels, axis=1),
                    tf.argmax(modules[-1], axis=1)
                )
            # tf.argmax(tf.nn.softmax(modules[-1]), axis=1)
            with tf.name_scope("accuracy"):
                accuracy = tf.reduce_mean(
                    tf.cast(correct_predict_labels, tf.float32),
                    name='accuracy'
                )
            with tf.name_scope("error"):
                error = tf.losses.mean_squared_error(
                    tf.argmax(labels, axis=1),
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
        print("\n=== Network Architecture ===")
        pprint.pprint(modules)
        print("=== Network Architecture ===\n")
        del W, b, modules
        return logits, probab, accuracy, loss, error, penlayer_output,\
            reg, train_op

    def train(self):
        # if tf.gfile.Exists("./" + self.network_type):
        #     tf.gfile.DeleteRecursively(f"./{self.network_type}")
        # dataset = tf.data.Dataset.from_tensor_slices((self.x, self.y))
        # dataset = dataset.repeat(self.epochs)
        # dataset = dataset.batch(self.batch_size)
        # iterator = dataset.make_initializable_iterator()
        epos = []
        batch_acc = []
        batch_loss = []
        batch_error = []
        batch_reg = []
        n_batches = self.data.train.len // self.batch_size
        for epoch in range(self.epochs):
            for mbatch in range(n_batches):
                epos.append(epoch)
                trX = self.data.train.images[
                    (mbatch*self.batch_size): self.batch_size * (mbatch+1)
                ]
                trY = self.data.train.labels[
                    (mbatch*self.batch_size): self.batch_size * (mbatch+1)
                ]
                train_data = np.c_[
                    trX.reshape(-1, np.prod(trX.shape[1:])),
                    trY
                ]
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
                            # self.summary,
                            # iterator.initializer
                        ],
                        feed_dict={
                            self.x: train_data
                            [:, :np.prod(trX.shape[1:])].reshape(trX.shape),
                            self.y: train_data
                            [:, np.prod(trX.shape[1:]):].reshape(trY.shape)
                        }
                    )
                batch_acc.append(acc)
                batch_loss.append(loss)
                batch_error.append(error)
                batch_reg.append(reg)
                # self.train_writer.add_summary(train_summary, epoch)
                self.train_writer.flush()

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
            for val in self.net['train']:
                self.run.log_scalar(
                    "training_{}".format(val),
                    self.net['train'][val][epoch],
                    epoch
                )
            ########################################################
            # validation                                           #
            ########################################################
            _ = self.predict(epoch, mode='valid')

            self.best_valid_loss = self.net['valid']['loss'][epoch - 1]
            if self.net['valid']['loss'][epoch] < self.best_valid_loss:
                self.best_valid_loss = self.net['valid']['loss'][epoch]
                dirname = f"{self.models}/{self.dataset}/{self.network_type}"
                fname = f"/epoch_{epoch}_loss_{self.best_valid_loss}"
                if not os.path.exists(dirname):
                    os.makedirs(dirname + fname)
                # self.saver.save(
                #     self.sess, dirname + fname, global_step=epoch
                # )
                self.run.info['ckpt'] = dirname + fname
                ############################################################
                # scatter on 1K train data                                 #
                ############################################################
                X = self.data.train.images[1600:2600]
                Y = self.data.train.labels[1600:2600]
                penultimate, probab = self.sess.run(
                    [self.l2, self.probabilities],
                    feed_dict={self.x: X, self.y: Y}
                )
                plot.plot_scatter(
                    self,
                    penultimate,
                    Y.argmax(axis=1),
                    probab.max(axis=1),
                    self.net['valid']['loss'][epoch],
                    epoch
                )
                for val in self.net['valid']:
                    self.run.log_scalar(
                        "validation_{}".format(val),
                        self.net['valid'][val][epoch],
                        epoch
                    )
                self.run.info['best_valid_loss'] = \
                    self.net['valid']['loss'][epoch]

            ########################################################
            # test                                                 #
            ########################################################
            if epoch > 0 and epoch % 25 == 0:
                _ = self.predict(epoch // 25 - 1, mode='test')
                self.run.log_scalar(
                    "test_accuracy",
                    self.net['test']['acc'][epoch // 25 - 1],
                    epoch // 25 - 1
                )
            #self.train_writer.add_summary(img_summary, epoch)
        self.train_writer.close()
        self.valid_writer.close()
        self.test_writer.close()
        # self.sess.close()
        # tf.reset_default_graph()
        # ops.reset_default_graph()
        return epos, self.net

    def predict(
            self,
            epoch=None,
            batch_xs=None,
            batch_ys=None,
            mode='valid',
            verbose=True
    ):
        n_samples = self.data.valid.len if mode == 'valid' \
            else self.data.test.len if mode == 'test' else len(batch_xs)

        logits = []
        labels = []
        batch_acc = []
        batch_loss = []
        batch_error = []
        batch_reg = []
        # this for loop is equivalent to one epoch, going through all data
        n_batches = n_samples // self.batch_size
        for mbatch in range(n_batches):
            if mode == 'valid' or mode == 'test':
                data = getattr(self.data, mode)
                X = data.images[
                    (mbatch*self.batch_size): self.batch_size * (mbatch + 1)
                ]
                Y = data.labels[
                    (mbatch*self.batch_size): self.batch_size * (mbatch + 1)
                ]
            else:
                X = batch_xs[
                    (mbatch*self.batch_size): self.batch_size * (mbatch+1)
                ]
                Y = batch_ys[
                    (mbatch*self.batch_size): self.batch_size * (mbatch+1)
                ]

            test_data = np.c_[X.reshape(-1, np.prod(X.shape[1:])), Y]
            # np.random.shuffle(test_data)
            logts, acc, loss, error, reg  = self.sess.run(
                [
                    self.logits,
                    self.accuracy,
                    self.loss,
                    self.error,
                    self.reg,
                    # self.summary,
                ],
                feed_dict={
                    self.x: test_data
                    [:, :np.prod(X.shape[1:])].reshape(X.shape),
                    self.y: test_data
                    [:, np.prod(X.shape[1:]):].reshape(Y.shape)
                }
            )
            logits.append(logts)
            # labels.append(Y)  # labels.append(test_data[:, np.prod(X.shape[1:]):].reshape(Y.shape))
            labels.append(test_data[:, np.prod(X.shape[1:]):].reshape(Y.shape))
            batch_acc.append(acc)
            batch_loss.append(loss)
            batch_error.append(error)
            batch_reg.append(reg)
            # if mode == 'valid':
            #     self.valid_writer.add_summary(summary, epoch)
            #     self.valid_writer.flush()
            # if mode == 'test':
            #     self.test_writer.add_summary(summary, epoch)
            #     self.test_writer.flush()

        self.net[mode]['acc'].append(np.array(batch_acc).mean())
        self.net[mode]['loss'].append(np.array(batch_loss).mean())
        self.net[mode]['error'].append(np.array(batch_error).mean())
        self.net[mode]['reg'].append(np.array(batch_reg).mean())
        if verbose:
            # if mode == 'test':
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


# if __name__ == "__main__":
#     if tf.gfile.Exists('./logs'):
#         tf.gfile.DeleteRecursively('./logs')
#     tf.gfile.MakeDirs('./logs/train/')
#     tf.gfile.MakeDirs('./logs/valid/')
#     tf.gfile.MakeDirs('./logs/test/')
#     nets = []
#     mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
#     ################################################################
#     # run resnet                                                   #
#     ################################################################
#     # nn = NeuralNetwork(mnist, 'resnet', epochs=10, batch_size=64)
#     # trX = mnist.train.images.reshape(-1, 28, 28, 1)
#     # trY = mnist.train.labels
#     # valX = mnist.validation.images.reshape(-1, 28, 28, 1)
#     # valY = mnist.validation.labels
#     # history = nn.resnet_model.fit(
#     #     trX,
#     #     trY,
#     #     batch_size=64,
#     #     epochs=nn.epochs,
#     #     validation_data=(valX, valY),
#     #     shuffle=True
#     # )
#     # val_logits = nn.resnet_model.predict(valX)
#     # nn.set_temperature(logits=val_logits, labels=valY)
#     # ------------------------------------------------------------
#     for net in [
#             "network_scatter_embedding", "network_vanilla", "network_l2",
#             "network_scatter_weights"
#     ]:
#         nn = NeuralNetwork(mnist, net, epochs=5, batch_size=64)
#         epochs, network = nn.train()
#         nets.append(network)
#     with open(f"./logs/{nn.dataset}_nets_logs.pickle", "wb") as fh:
#         pickle.dump(nets, fh)
