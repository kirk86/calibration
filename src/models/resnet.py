from __future__ import absolute_import

import tensorflow as tf
L = tf.keras.layers
K = tf.keras.backend
# InputSpec = tf.keras.layers.InputSpec


class Residual(tf.keras.layers.Wrapper):
    """This wrapper automatically applies a residual to a model.

    For an input `x` and a model `F(x)`, the residual wrapper gives the output
    `y = x + F(x)`. In this configuration, the output of F(x) must have the
    same shape as x. Other merge modes are supported besides summation.

    ```python
        input = Input(shape=(5,))

        # Apply the residual normally
        output1 = Residual(Dense(5), merge_mode='sum')(input)

        # Throws an exception due to mismatching shapes
        output2 = Residual(Dense(3), merge_mode='sum')(input)

        # Product: `y = x * F(x)`
        output3 = Residual(Dense(5), merge_mode='mul')(input)
    ```

    For more modes, see: https://keras.io/layers/core/#merge

    Alternatively, a function which takes the input and the layer output
    can be passed to define the merge:

    ```python
        from keras.layers import Merge
        def diff_merge():  # x_fx = [x, fx]
            diff = lambda x: x[1] - x[0]
            return Merge(mode=diff, output_shape=lambda x: x[0])

        # Difference: `y = F(x) - x`
        output4 = Residual(Dense(5), merge_mode=diff_merge())(input)
    ```

    Arguments:
        layer: The layer to wrap
        merge_mode: The merge operation
    """

    def __init__(self, layer, merge_mode='sum', **kwargs):
        self.merge_mode = merge_mode
        self.supports_masking = True
        super(Residual, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        # output_shape = self.layer.get_output_shape_for(input_shape)
        output_shape = self.get_output_shape_for(input_shape)
        if output_shape != input_shape:
            raise Exception(
                'Cannot apply residual to layer "{}": '
                'mismatching input and output shapes'
                '"{}" and "{}"'.format(
                    self.layer.name, input_shape, output_shape
                )
            )
        if not self.layer.built:
            self.layer.build(input_shape)
            self.layer.built = True
        self.input_spec = [L.InputSpec(shape=input_shape)]
        super(Residual, self).build()

    def get_output_shape_for(self, input_shape):
        return input_shape

    def call(self, x, mask=None):
        # layer_output = self.layer.call(x, mask=mask)
        layer_output = self.layer.call(x)
        if isinstance(self.merge_mode, str) and self.merge_mode == 'sum':
            # self.merge_mode = tf.keras.layers.Merge(mode=self.merge_mode)
            self.merge_mode = L.Add()
        output = self.merge_mode([x, layer_output])
        return output

    @classmethod
    def from_config(cls, config):
        from tf.keras.utils.utils import deserialize_keras_object
        merge_mode = deserialize_keras_object(config.pop('merge_mode'))
        residual = super(Residual, cls).from_config(config)
        residual.merge_mode = merge_mode
        return residual

    def get_config(self):
        config = {
            "merge_mode": {
                'class_name': 'Merge',
                'config': self.merge_mode.get_config()
            }
        }
        base_config = super(Residual, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
