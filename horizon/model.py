import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2, mobilenet_v2, ResNet50
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2



def backbone(input_shape, x):
    #backbone = MobileNetV2(input_shape=input_shape, include_top=False, input_tensor=x, pooling="max", alpha=0.5)
    backbone = ResNet50(input_shape=input_shape, include_top=False, input_tensor=x)
    ftr_layer = "conv5_block3_2_relu"

    features = backbone.get_layer(ftr_layer).output
    features = GlobalAvgPool2D()(features)
    features = Flatten(name="features")(features)
    features = Dense(256, activation="relu")(features)

    return features


def horizon_global_context(input_shape=(224,224,3), t_bins=500, r_bins=500):
    """
    RGB image (224,224,3)
    """
    image_in = Input(shape=input_shape)
   
    features = backbone(input_shape, image_in)
    
    # Theta head
    y = Dense(256, activation="relu")(features)
    y = Dropout(0.1)(y)
    y = Dense(256, activation="relu")(y)
    theta_bins = Dense(t_bins, activation="softmax", name="theta", kernel_regularizer=None)(y)

    # Rho head
    y = Dense(256, activation="relu")(features)
    y = Dropout(0.1)(y)
    y = Dense(256, activation="relu")(y)
    rho_bins = Dense(r_bins, activation="softmax",name="rho", kernel_regularizer=None)(y)

    # The model
    return Model(image_in, [theta_bins, rho_bins] )

