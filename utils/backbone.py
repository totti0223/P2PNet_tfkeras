import tensorflow as tf
from tensorflow.keras.applications import *
from loguru import logger

# Define the dictionary mapping model names to model functions
MODEL_MAP = {
    'vgg16': {"core": VGG16,
            "preprocess_input": vgg16.preprocess_input,
            "intermediate_layer_names": ["block4_conv3", "block5_conv3"]}, # ここも1/8to1/16をつかう
    'mobilenetv3_small': {
        "core": MobileNetV3Small,
        "preprocess_input": None,
        "intermediate_layer_names": None
    },
    'mobilenetv3_large': {
        "core":MobileNetV3Large,
        "preprocess_input": None,
        #"intermediate_layer_names": ["re_lu_2", "re_lu_6", "multiply_1","multiply_13","multiply_19"] #1/2, 1/4, 1/8, 1/16, 1/32# 320,160,80,40,20
        #"intermediate_layer_names": ["multiply_1","multiply_13","multiply_19"] #1/8, 1/16, 1/32# 320,160,80,40,20
        "intermediate_layer_names": ["multiply_1","multiply_13","multiply_19"] #1/16, 1/32# 320,160,80,40,20
        }
    }
def load_feature_extraction_model(model_name=None, input_shape=(None,None,3), preprocessing=True, name = "backbone"):
    logger.info(f"Loading model: {model_name}")
    if model_name not in MODEL_MAP:
        raise ValueError(f"Invalid model name: {model_name}")
    model_map = MODEL_MAP[model_name]
    model_fn = model_map["core"]
    layers_names = model_map["intermediate_layer_names"]
    logger.info(model_map)
    
    backbone = select_backbone(model_fn, input_shape, preprocessing)   
    
    feature_extraction_model = tf.keras.Model(inputs=backbone.input, outputs=[backbone.get_layer(layer).output for layer in layers_names], name = name)

    # add preprocessing layer if needed
    if model_map["preprocess_input"] is not None and preprocessing:
        inputs = tf.keras.layers.Input(shape=input_shape)
        preprocessed_inputs = tf.keras.layers.Lambda(model_map["preprocess_input"], "preproc", name="preproc")(inputs)
        outputs = feature_extraction_model(preprocessed_inputs)
        feature_extraction_model = tf.keras.Model(inputs=inputs, outputs=outputs, name = name)
        
    return feature_extraction_model

def select_backbone(model_fn, input_shape, preprocessing):    
    # Check if class has the attribute include_preprocessing
    # due to keras name space issue, cannnot use hasattr currently. 
    try:
        model = model_fn(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=None,
            include_preprocessing=preprocessing,
            )
    except Exception as e:
        model = model_fn(
            include_top=False,
            weights='imagenet',
            input_shape=input_shape,
            pooling=None,
            )
        
    return model