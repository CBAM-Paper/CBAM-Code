import copy
import logging
from pathlib import Path
from timeit import default_timer as timer
from typing import List, Dict
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

from raopt.ml.tensorflow_preamble import TensorflowConfig

TensorflowConfig.configure_tensorflow()

from tensorflow.keras.layers import GlobalAveragePooling1D, Reshape, Dense, Multiply

import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Bidirectional, \
    Masking, Concatenate, TimeDistributed, LSTM
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization, Dense, Dropout, Add

from raopt.ml.encoder import encode_trajectory, decode_trajectory
from raopt.ml.loss import euclidean_loss
from raopt.utils.config import Config
from tensorflow.keras import layers, models  # 增加 models
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Input, Dense, GRU, TimeDistributed, Concatenate, MultiHeadAttention, Masking, \
    Bidirectional, LSTM, LayerNormalization, Add
from tensorflow.keras.models import Model
import logging

ADD_REF_POINT_IN_MODEL = False
SCALE_IN_MODEL = True
param_path = Config.get_parameter_path()
Path(param_path).mkdir(exist_ok=True)
LEARNING_RATE = Config.get_learning_rate()
MODEL_NAME = 'RAoPT'
FEATURES = ['latlon', 'hour', 'dow']
log = logging.getLogger()


def _encode(x: List[pd.DataFrame]) -> np.ndarray:
    encodings = [encode_trajectory(t) for t in tqdm(x, total=len(x), desc='Encoding', leave=False)]
    return encodings


def _decode(x: np.ndarray, originals: List[pd.DataFrame], ignore_time: bool = True) -> List[pd.DataFrame]:

    if 'taxi_id' in originals[0]:
        uid = 'taxi_id'
    else:
        uid = 'uid'

    decodings = []
    for i, t in tqdm(enumerate(x), leave=False, desc='Decoding Trajectories', total=len(x)):
        decoded = decode_trajectory(t, ignore_time=ignore_time)
        decoded['trajectory_id'] = originals[i]['trajectory_id'][0]
        decoded['uid'] = originals[i][uid][0]
        if 'timestamp' in originals[i]:
            decoded['timestamp'] = originals[i]['timestamp']
        decodings.append(decoded)
    return decodings


class AttackModel:

    def __init__(
            self,
            reference_point: (float, float),
            scale_factor: (float, float),
            max_length: int,
            features: List[str] = FEATURES,
            vocab_size: Dict[str, int] = None,
            embedding_size: Dict[str, int] = None,
            learning_rate: float = LEARNING_RATE,
            parameter_file: str = None,
    ):
        self.history = None
        self.max_length = max_length
        self.features = features
        assert self.features[0] == 'latlon'
        if vocab_size is None:
            self.vocab_size = {
                'latlon': 2,
                'hour': 24,
                'dow': 7,
            }
        else:
            self.vocab_size = vocab_size
        if embedding_size is None:
            self.embedding_size = {
                'latlon': 64,
                'hour': 24,
                'dow': 7,
            }
        else:
            self.embedding_size = embedding_size
        self.num_features = sum(self.vocab_size[k] for k in features)
        self.scale_factor = scale_factor
        self.lat0, self.lon0 = reference_point
        self.param_file = parameter_file

        self.optimizer = Adam(learning_rate)  # Good default choice

        self.model = self.build_model()

        loss = euclidean_loss if self.scale_factor[0] < 90 else 'mse'
        self.model.compile(
            loss=loss,
            optimizer=self.optimizer,
        )

    def build_model(self) -> Model:

        inputs = Input(shape=(self.max_length, self.num_features),
                       name="Input_Encoding")

        masked = Masking(name="Mask_Padding")(inputs)

        split_points = [
            self.vocab_size[feature]
            for feature in self.features

        ]
        in_elements = tf.split(
            masked, split_points,
            axis=-1,
            name="split_features"
        )

        embeddings = []
        for i, feature in enumerate(self.features):
            emb = Dense(units=self.embedding_size[feature],
                        activation='relu',
                        name=f'Embedding_{feature}_dense')
            embedding = TimeDistributed(emb, name=f'Embedding_{feature}')(in_elements[i])
            embeddings.append(embedding)

        if len(embeddings) > 1:
            concatenation = Concatenate(axis=-1, name="Join_Features")(embeddings)
        else:
            concatenation = embeddings[0]
        feature_fusion = Dense(
            units=128,
            activation='relu',
            name='Feature_Fusion'
        )(concatenation)
        # ----------------------------------------------------------------------

        conv1 = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='Conv1D_Layer1')(
            feature_fusion)
        conv2 = Conv1D(filters=64, kernel_size=5, padding='same', activation='relu', name='Conv1D_Layer2')(conv1)
        conv3 = Conv1D(filters=64, kernel_size=7, padding='same', activation='relu', name='Conv1D_Layer3')(conv2)

        # Bidirectional LSTM layer ---------------------------------------------
        bidirectional_lstm_layer1_output = \
            Bidirectional(
                LSTM(units=128,
                     return_sequences=True,
                     ),
                name="Bidirectional_LSTM_Layer1",
            )(conv3)

        bidirectional_lstm_layer2_output = \
            Bidirectional(
                LSTM(units=64,
                     return_sequences=True,
                     ),
                name="Bidirectional_LSTM_Layer2",
            )(bidirectional_lstm_layer1_output)
        # ----------------------------------------------------------------------
        attention_output = MultiHeadAttention(
            num_heads=8,
            key_dim=64,  # 每个注意力头的维度，通常设置为embedding_dim/num_heads
            name="MultiHead_Attention"
        )(bidirectional_lstm_layer2_output, bidirectional_lstm_layer2_output)

        # Output Layer ---------------------------------------------------------
        output_lat = TimeDistributed(
            Dense(1, activation='tanh'), name='Output_lat')(attention_output)
        output_lon = TimeDistributed(
            Dense(1, activation='tanh'), name='Output_lon')(attention_output)
        if SCALE_IN_MODEL:
            offset = (self.lat0, self.lon0) if ADD_REF_POINT_IN_MODEL else (0., 0.)
            lat_scaled = TimeDistributed(Rescaling(scale=self.scale_factor[0], offset=offset[0]),
                                         name='Output_lat_scaled')(output_lat)
            lon_scaled = TimeDistributed(Rescaling(scale=self.scale_factor[1], offset=offset[1]),
                                         name='Output_lon_scaled')(output_lon)
            outputs = [lat_scaled, lon_scaled]
        else:
            outputs = [output_lat, output_lon]

        if len(outputs) > 1:
            output = Concatenate(axis=-1, name="Output_Concatenation")(outputs)
        else:
            output = outputs

        model = Model(inputs=inputs, outputs=output,
                      name=MODEL_NAME)

        # Save Model-----------------------------------------------------------
        file = param_path + f"{MODEL_NAME}.json"
        with open(file, 'w') as f:
            f.write(model.to_json())
        # ----------------------------------------------------------------------

        return model

    # 填充轨迹
    def preprocess_x(self, x: np.ndarray) -> np.ndarray:

        x = copy.deepcopy(x)

        for i in range(len(x)):
            x[i][:, 0] -= self.lat0
            x[i][:, 1] -= self.lon0

        x = pad_sequences(
            x, maxlen=self.max_length, padding='pre', dtype='float64'
        )

        return x

    def preprocess_y(self, y: np.ndarray) -> np.ndarray:

        y = copy.deepcopy(y)

        if not ADD_REF_POINT_IN_MODEL:
            for i in range(len(y)):
                y[i][:, 0] -= self.lat0
                y[i][:, 1] -= self.lon0

        y = pad_sequences(
            y, maxlen=self.max_length, padding='pre', dtype='float64'
        )

        if not SCALE_IN_MODEL:
            y[:, :, 0] /= self.scale_factor[0]
            y[:, :, 1] /= self.scale_factor[1]


        y = y[:, :, :2]

        return y

    def postprocess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        if not SCALE_IN_MODEL:
            y[:, :, 0] *= self.scale_factor[0]
            y[:, :, 1] *= self.scale_factor[1]

        if not ADD_REF_POINT_IN_MODEL:
            y[:, :, 0] += self.lat0
            y[:, :, 1] += self.lon0

        result = []
        for i in range(len(y)):
            n = len(y[i]) - len(x[i])
            result.append(y[i][n:])
        return result

    def train(self, x: np.ndarray,
              y: np.ndarray,
              epochs: int = Config.get_epochs(),
              batch_size: int = Config.get_batch_size(),
              val_x: np.ndarray = None,
              val_y: np.ndarray = None,
              tensorboard: bool = False,
              use_val_loss: bool = False,
              early_stopping: int = Config.get_early_stop()
              ):

        x_train = self.preprocess_x(x)

        y_train = self.preprocess_y(y)

        if val_x is None:
            validation_data = None
        else:
            val_x = self.preprocess_x(val_x)
            val_y = self.preprocess_y(val_y)
            validation_data = (val_x, val_y)
        if use_val_loss:
            validation_split = 0.1
            stop_metric = 'val_loss'
        else:
            validation_split = 0.0
            stop_metric = 'loss'

        if self.param_file is None:
            checkpoint_path = param_path + \
                              MODEL_NAME + "_epoch{epoch:03d}.hdf5"
        else:
            checkpoint_path = self.param_file
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor=stop_metric,
            save_best_only=True,
            mode='min',
            save_weights_only=True,
            verbose=0,
            save_freq='epoch'
        )

        callbacks = [
            TqdmCallback(verbose=1, leave=False),
            checkpoint,
        ]

        if early_stopping > 0:
            callbacks.append(
                EarlyStopping(
                    monitor=stop_metric,
                    patience=early_stopping,
                    mode='min',
                    restore_best_weights=True,
                    verbose=0
                )
            )

        # Tensorboard
        if tensorboard:
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=Config.get_tensorboard_dir())
            callbacks.append(tensorboard_callback)
        # 在 train 方法中，model.fit() 会自动执行前向传播和反向传播。
        # 通过计算每个批次的损失，Keras 会通过反向传播调整模型的权重，以最小化损失函数。
        self.history = self.model.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_split=validation_split
        )

        return self.history

    def _predict(self, x: np.ndarray) -> np.ndarray:
        x_test = self.preprocess_x(x)

        prediction = self.model.predict(x_test)

        return self.postprocess(x, prediction)

    def predict(self, x: np.ndarray or list or pd.DataFrame) -> np.ndarray or List[pd.DataFrame] or pd.DataFrame:

        if type(x) is pd.DataFrame or len(x[0]) == self.num_features:
            x = [x]
            single = True
        else:
            single = False

        if type(x[0]) is np.ndarray:
            result = self._predict(x)
        elif type(x[0]) is pd.DataFrame:
            start = timer()
            encoded = _encode(x)
            log.info(f"Encoded trajectories in {round(timer() - start)}s")

            start = timer()
            prediction = self._predict(encoded)
            log.info(f"Prediction in {round(timer() - start)}s")

            start = timer()
            decoded = _decode(prediction, x)
            log.info(f"Decoded trajectories in {round(timer() - start)}s")
            result = decoded
        else:
            log.error(f"Unexpected input type {type(x[0])}")
            raise ValueError(f"Unexpected input type {type(x[0])}")

        if single:
            return result[0]
        else:
            return result

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> dict:
        x_test = self.preprocess_x(x)
        y_test = self.preprocess_y(y)

        d = self.model.evaluate(x_test, y_test, return_dict=True)
        return d
