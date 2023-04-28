import tensorflow as tf
from .backbone import load_feature_extraction_model
import numpy as np

class RefPoints(tf.keras.layers.Layer):
    def __init__(self, pyramid_levels=None, row=2, line=2):
        super(RefPoints, self).__init__()
        self.pyramid_levels = pyramid_levels
        self.strides = [2 ** x for x in self.pyramid_levels]
        self.row = row
        self.line = line

    # generate the reference points in grid layout
    def generate_ref_points(self, stride=8, row=2, line=2):
        row_step = stride / row
        line_step = stride / line
        x = (tf.range(1, line + 1, dtype=tf.float32) - 0.5) * line_step - stride / 2
        y = (tf.range(1, row + 1, dtype=tf.float32) - 0.5) * row_step - stride / 2
        x, y = tf.meshgrid(x, y)
        ref_points = tf.stack((
            tf.reshape(x, (-1,)),
            tf.reshape(y, (-1,))
        ), axis=-1)

        return ref_points

    def add_offset2refpoints(self, shape, stride, ref_points):
        x = (tf.range(0, shape[1], dtype=tf.float32) + 0.5) * stride
        y = (tf.range(0, shape[0], dtype=tf.float32) + 0.5) * stride
        x, y = tf.meshgrid(x, y)
        shifts = tf.stack([tf.reshape(x, [-1]), tf.reshape(y, [-1])], axis=1)
        M = tf.shape(ref_points)[0]
        N = tf.shape(shifts)[0]
        offsetted_ref_points = (tf.reshape(ref_points, [1, M, 2]) + tf.reshape(shifts, [N, 1, 2]))
        offsetted_ref_points = tf.reshape(offsetted_ref_points, [M * N, 2])
        return offsetted_ref_points

    def call(self, image):
        image_shape = tf.shape(image)[1:3]
        image_shapes = [(image_shape + 2 ** x - 1) // (2 ** x) for x in self.pyramid_levels]
        output_ref_points = tf.zeros((0, 2), dtype=tf.float32)
        
        ref_points = self.generate_ref_points(2**self.pyramid_levels[0], row=self.row, line=self.line)
        offsetted_ref_points = self.add_offset2refpoints(image_shapes[0], self.strides[0], tf.cast(ref_points, tf.float32))
        output_ref_points = tf.concat([output_ref_points, offsetted_ref_points], axis=0)
        
        output_ref_points = tf.tile(
            tf.expand_dims(output_ref_points, axis=0),
            multiples = [tf.shape(image)[0], 1, 1]
        )
        return output_ref_points
    
    def get_config(self):
        config = super(RefPoints, self).get_config()
        config.update({
            'pyramid_levels': self.pyramid_levels,
            'row': self.row,
            'line': self.line,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class P2PNet(tf.keras.models.Model):
    def __init__(self,
                 input_shape=(None, None, 3),
                 backbone_name = "mobilenetv3_large",
                 preprocessing=True,
                 feature_size = 512,
                 no_reference_points = 4,
                 no_classes = 1,
                #  optimizer=tf.keras.optimizers.Adam(1e-5),
                 gamma=100,
                  **kwargs):
        super(P2PNet, self).__init__(**kwargs)
        
        #self.loss_func = custom_loss
        self.no_classes = no_classes + 1
        self.feature_size = feature_size
        self.gamma = gamma
        #self.input_tensor = tf.keras.layers.Input([None, None, 3], batch_size=None, dtype=tf.float32)
        self.no_reference_points = no_reference_points
        self.backbone_name = backbone_name
        self.inputshape = input_shape
        self.preprocessing = preprocessing
        assert self.no_reference_points in [1, 4, 9], "must be 4 9 16,,,"

        # prepare layers
        #self.inputs = tf.keras.layers.Input(shape=self.inputshape)
        self.backbone = load_feature_extraction_model(self.backbone_name,
                                                      self.inputshape,
                                                      self.preprocessing,
                                                      name = "backbone")
        self.feature_output_number = len(self.backbone.outputs)
        #print(self.feature_output_number)
        # fpn
        self.fpn_conv0 = tf.keras.layers.Conv2D(self.feature_size, 1, padding="same", name="fpn_conv0")
        for i in range(1, self.feature_output_number):
            #print("set",i)
            setattr(self, "fpn_upsample%d" % i, tf.keras.layers.UpSampling2D((2,2), name="fpn_upsample%d" % i))
            setattr(self, "fpn_conv%d" % i, tf.keras.layers.Conv2D(self.feature_size, 1, padding="same", name="fpn_conv%d" % i))
            setattr(self, "fpn_add%d" % i, tf.keras.layers.Add(name="fpn_add%d" % i))
        self.fpn_convout = tf.keras.layers.Conv2D(self.feature_size, 3, padding="same", name="fpn_convout%d" % i)
        
        # classification head
        self.cls_head0 = tf.keras.layers.Conv2D(self.feature_size, 3, padding="same", name="cls_head0")
        self.cls_relu0 = tf.keras.layers.ReLU(name="cls_relu0")
        self.cls_head1 = tf.keras.layers.Conv2D(self.feature_size, 3, padding="same", name="cls_head1")
        self.cls_relu1 = tf.keras.layers.ReLU(name="cls_relu1")
        self.cls_head2 = tf.keras.layers.Conv2D(self.no_classes * self.no_reference_points, 3, padding='same', name="clshead2")

        # regression head
        self.reg_head0 = tf.keras.layers.Conv2D(self.feature_size, 3, padding="same", name="reg_head0")
        self.reg_relu0 = tf.keras.layers.ReLU(name="reg_relu0")
        self.reg_head1 = tf.keras.layers.Conv2D(self.feature_size, 3, padding="same", name="reg_head1")
        self.reg_relu1 = tf.keras.layers.ReLU(name="reg_relu1")
        self.reg_head2 = tf.keras.layers.Conv2D(2 * self.no_reference_points, 3, padding="same", name="reg_head2")

    def call(self, inputs):
        intermediate_outputs = self.backbone(inputs)
        x = intermediate_outputs[-1]  # the last element is the deepest(smallest) feature map
        x = self.fpn_conv0(x)
        # for i in range(self.feature_output_number - 1, -1, -1):
        for i, feature in enumerate(intermediate_outputs[-2::-1]):
            #print("get",i)
            x = getattr(self, "fpn_upsample%d" % (i+1))(x)
            _x = getattr(self, "fpn_conv%d" % (i+1))(feature)
            x = getattr(self, "fpn_add%d" % (i+1))([x, _x])
        fpn = self.fpn_convout(x)

        #classification
        cls = self.cls_head0(fpn)
        cls = self.cls_relu0(cls)
        cls = self.cls_head1(cls)
        cls = self.cls_relu1(cls)
        cls = self.cls_head2(cls)
        clsshape = tf.shape(cls)
        cls = tf.reshape(cls, (clsshape[0], clsshape[1], clsshape[2], self.no_classes, self.no_reference_points))
        cls = tf.reshape(cls, (clsshape[0], -1, self.no_classes), name = "classification_head_output")

        #regression
        reg = self.reg_head0(fpn)
        reg = self.reg_relu0(reg)
        reg = self.reg_head1(reg)
        reg = self.reg_relu1(reg)
        reg = self.reg_head2(reg)
        regshape = tf.shape(reg)
        reg = tf.reshape(reg, (regshape[0], -1, 2))

        # anchor points
        ref_points = RefPoints(
            pyramid_levels=[3,],
            row=np.sqrt(self.no_reference_points),
            line=np.sqrt(self.no_reference_points))(inputs)
    
        reg = reg * self.gamma + ref_points

        outputs = tf.concat([reg, cls], axis=-1)
        return outputs

    def train_step(self, data):
        # Unpack the data
        x, y_true = data

        # Record the forward and backward pass in a GradientTape
        with tf.GradientTape() as tape:
            # Forward pass
            y_pred = self(x, training=True)

            # Compute the loss value
            loss = self.compiled_loss(y_true, y_pred)
        # Compute the gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # Apply the gradients
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # Update the metrics
        self.compiled_metrics.update_state(y_true, y_pred)

        # Return a dictionary mapping metric names to their values
        results = {m.name: m.result() for m in self.metrics}
        return results
    
    def test_step(self, data):
        x, y_true = data
        y_pred = self(x, training=False)
        loss = self.compiled_loss(y_true, y_pred)  # Replace custom_loss with your actual loss function
        self.compiled_metrics.update_state(y_true, y_pred)

        # Return a dictionary mapping metric names to their values
        results = {m.name: m.result() for m in self.metrics}
        return results
    
    # def build(self, input_shape):
    #     input_layer = tf.keras.layers.Input(shape=input_shape[1:])
    #     _ = self.call(input_layer)
    #     super(P2PNet, self).build(input_shape)
    
    def build_graph(self):
        x = tf.keras.layers.Input(shape=(128, 128, 3))
        return tf.keras.models.Model(inputs=[x], outputs=self.call(x))
    
    def get_config(self):
        config = super(P2PNet, self).get_config()
        config.update({
            'input_shape': self.inputshape,
            'backbone_name': self.backbone_name,
            'preprocessing': self.preprocessing,
            'feature_size': self.feature_size,
            'no_reference_points': self.no_reference_points,
            'no_classes': self.no_classes - 1,
            'gamma': self.gamma,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)