"""
### reference
- https://keras.io/keras_core/api/layers/attention_layers/multi_head_attention/
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    LayerNormEps = 1e-6
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=LayerNormEps)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
        )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=LayerNormEps)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_multivariate_transformer_model( # multivariate transformer model for gait forecasting
    look_back, n_features,
    num_outputs,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units, 
    dropout=0,
    mlp_dropout=0,
    ):
    
    # encoder
    inputs = keras.Input(shape=(look_back, n_features)) #shape=input_shape->(look_back, n_features)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    # decoder
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu",
                         kernel_initializer='random_normal',
                         bias_initializer='zeros')(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    timeseries_outputs = layers.Dense(num_outputs, activation="relu", #linear->check!
                                      kernel_initializer='random_normal',
                                      bias_initializer='zeros',
                                      name='ts_output')(x) 
    return keras.Model(inputs, timeseries_outputs)


def build_multivariate_transformer_classification_model( # multivariate transformer model for classification
    look_back, n_features,
    num_classes,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units, 
    dropout=0,
    mlp_dropout=0,
    ):
    
    #encoder
    inputs = keras.Input(shape=(look_back, n_features))
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    
    #decoder
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu",
                         kernel_initializer='random_normal',
                         bias_initializer='zeros')(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    classification_outputs = layers.Dense(num_classes, activation="softmax",  # softmax for classification
                                          kernel_initializer='random_normal',
                                          bias_initializer='zeros',
                                          name='classification_output')(x)
    return keras.Model(inputs, classification_outputs)


