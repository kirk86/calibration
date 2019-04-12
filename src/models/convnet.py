import tensorflow as tf
keras = tf.keras
KM = tf.keras.models
KL = tf.keras.layers
KO = tf.keras.optimizers


def build_model():
    model = KM.Sequential()
    model.add(
        KL.Conv2D(6, (5, 5), padding='valid', activation = 'relu',
                  kernel_initializer='he_normal', input_shape=(32, 32, 3)))
    model.add(KL.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(
        KL.Conv2D(16, (5, 5), padding='valid', activation = 'relu',
                  kernel_initializer='he_normal'))
    model.add(KL.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(KL.Flatten())
    model.add(
        KL.Dense(120, activation = 'relu', kernel_initializer='he_normal'))
    model.add(
        KL.Dense(84, activation = 'relu', kernel_initializer='he_normal'))
    model.add(
        KL.Dense(10, activation = 'softmax', kernel_initializer='he_normal'))
    sgd = KO.SGD(lr=0.14, momentum=0.9, nesterov=True)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model


def lenet(self, inputs, outputs, kernel, pool):
    conv1 = tf.layers.Conv2D(
        6, (5, 5), padding='valid', activation='relu',
        kernel_initializer='he_normal'
    )(inputs)
    pool1 = tf.layers.MaxPooling2D(pool_size=pool, strides=2)(conv1)
    conv2 = tf.layers.Conv2D(
        16, (5, 5), padding='valid', activation='relu',
        kernel_initializer='he_normal'
    )(pool1)
    pool2 = tf.layers.MaxPooling2D(pool_size=pool, strides=2)(conv2)
    flat = tf.layers.Flatten()(pool2)
    dense1 = tf.layers.Dense(120, activation='relu',
                             kernel_initializer='he_normal')(flat)
    dense2 = tf.layers.Dense(84, activation='relu',
                             kernel_initializer='he_normal')(dense1)
    dense3 = tf.layers.Dense(10, activation=tf.identity,
                             kernel_initializer='he_normal')(dense2)

    scaling_factor = tf.constant(1.) / self.batch_size
    C = tf.eye(self.batch_size) - scaling_factor * tf.ones(
        shape=[self.batch_size, self.batch_size])  # 64x64
    S = tf.matmul(tf.matmul(dense3, C, transpose_a=True),
                  dense1)  # 2x2
    reg = tf.abs(tf.reduce_sum(S)) * self.regularization
    loss = self.objective(
        labels=self.y, logits=dense3, name='softmax_v2') + reg

    with tf.name_scope("train"):
        # train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        train_op = self.optimizer(self.learning_rate).minimize(loss)
    with tf.name_scope("metrics"):
        with tf.name_scope("correctly_predicted_labels"):
            correct_predict_labels = tf.equal(
                tf.argmax(outputs, axis=1),
                tf.argmax(tf.nn.softmax(dense3), axis=1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(
                tf.cast(correct_predict_labels, tf.float32),
                name='accuracy')
        with tf.name_scope("error"):
            error = tf.reduce_mean(
                tf.squared_difference(
                    tf.argmax(outputs, axis=1),
                    tf.argmax(tf.nn.softmax(dense3), axis=1)),
                name='error')
        loss = tf.reduce_mean(loss, name='final_loss')
    tf.summary.scalar('cross_entropy_objective', loss)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("error", error)
    if not self.network_type == "vanilla":
        tf.summary.scalar("regularization_value", reg)
    logits = dense2
    probab = tf.nn.softmax(dense2)
    layer2_output = dense1
    return logits, probab, accuracy, loss, error, layer2_output, reg, \
        train_op


def convnet(self, inputs, outputs, kernel, pool):
    conv1 = tf.layers.Conv2D(
        32, kernel, padding='same', activation='relu'
    )(inputs)
    conv2 = tf.layers.Conv2D(
        32, kernel, padding='same', activation='relu'
    )(conv1)
    pool1 = tf.layers.MaxPooling2D(pool_size=pool, strides=1)(conv2)
    drop1 = tf.layers.Dropout(0.25)(pool1)
    conv3 = tf.layers.Conv2D(
        64, kernel, padding='same', activation='relu')(drop1)
    conv4 = tf.layers.Conv2D(
        64, kernel, padding='same', activation='relu')(conv3)
    pool2 = tf.layers.MaxPooling2D(pool_size=pool, strides=1)(conv4)
    drop2 = tf.layers.Dropout(0.25)(pool2)
    flat = tf.layers.Flatten()(drop2)
    dense1 = tf.layers.Dense(512, activation='relu')(flat)
    # drop3 = L.Dropout(0.5)(dense1)
    dense2 = tf.layers.Dense(10, activation='linear')(dense1)

    scaling_factor = tf.constant(1.) / self.batch_size
    C = tf.eye(self.batch_size) - scaling_factor * tf.ones(
        shape=[self.batch_size, self.batch_size])  # 64x64
    S = tf.matmul(tf.matmul(dense1, C, transpose_a=True),
                  dense1)  # 2x2
    reg = tf.abs(tf.reduce_sum(S)) * self.regularization
    reg = tf.constant(0.0)
    loss = self.objective(
        labels=self.y, logits=dense2, name='softmax_v2')

    with tf.name_scope("train"):
        # train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        train_op = self.optimizer(self.learning_rate).minimize(loss)
    with tf.name_scope("metrics"):
        with tf.name_scope("correctly_predicted_labels"):
            correct_predict_labels = tf.equal(
                tf.argmax(outputs, axis=1),
                tf.argmax(tf.nn.softmax(dense2), axis=1))
        with tf.name_scope("accuracy"):
            accuracy = tf.reduce_mean(
                tf.cast(correct_predict_labels, tf.float32),
                name='accuracy')
        with tf.name_scope("error"):
            error = tf.reduce_mean(
                tf.squared_difference(
                    tf.argmax(outputs, axis=1),
                    tf.argmax(tf.nn.softmax(dense2), axis=1)),
                name='error')
        loss = tf.reduce_mean(loss, name='final_loss')
    tf.summary.scalar('cross_entropy_objective', loss)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.scalar("error", error)
    if not self.network_type == "vanilla":
        tf.summary.scalar("regularization_value", reg)
    logits = dense2
    probab = tf.nn.softmax(dense2)
    layer2_output = dense1
    return logits, probab, accuracy, loss, error, layer2_output, reg, \
        train_op
