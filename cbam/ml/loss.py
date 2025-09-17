import numpy as np
import tensorflow as tf

EARTH_RADIUS = 6371008.8  # [meter]

# 角度转为弧度 计算地理坐标（经纬度）时的基本步骤
def degrees_to_radians(deg):
    """Convert degrees into radians with tensorflow methods."""
    return tf.constant(np.pi) / tf.constant(180, dtype=tf.float32) * deg

#计算两组经纬度坐标之间的 Haversine 距离，该距离是测量地球上两点之间的距离的常用方法，尤其适用于经纬度数据。
def haversine_distance_tf(y_true, y_pred):
    lat_true, lon_true = y_true[:, :, 0], y_true[:, :, 1]
    lat_pred, lon_pred = y_pred[:, :, 0], y_pred[:, :, 1]

    # convert all latitudes/longitudes from decimal degrees to radians
    lat1 = degrees_to_radians(lat_true)
    lon1 = degrees_to_radians(lon_true)
    lat2 = degrees_to_radians(lat_pred)
    lon2 = degrees_to_radians(lon_pred)

    lat = lat2 - lat1
    lng = lon2 - lon1
    # 计算公式 d 为两点之间的角距离。
    d = (tf.math.pow(tf.math.sin(lat * tf.constant(0.5)), 2)
         + tf.math.cos(lat1) * tf.math.cos(lat2) * tf.math.pow(tf.math.sin(lng * tf.constant(0.5)), 2))
    # 使用 tf.where 来确保计算中的角距离不会超出合理范围
    # The protection mechanisms might lead to coordinates that are invalid
    d = tf.where(d > 1.0, tf.ones_like(d), d)
    d = tf.where(d < 0.0, tf.zeros_like(d), d)

    x = tf.constant(2, dtype=tf.float32) * tf.constant(EARTH_RADIUS) * tf.math.asin(tf.math.sqrt(d))
    return x

# 欧几里得损失函数
@tf.autograph.experimental.do_not_convert
def euclidean_loss(y_true, y_pred):
    # 调用 haversine_distance_tf 计算每个轨迹点之间的 Haversine 距离（即预测轨迹与真实轨迹之间的距离）。
    # 然后，使用 mae（Mean Absolute Error）来计算每个轨迹的平均误差。mask 会确保忽略无效的轨迹点
    mask_value = tf.constant([0.0, 0.0], dtype=tf.float32)
    mask = tf.cast(tf.keras.backend.all(tf.keras.backend.not_equal(y_true, mask_value), axis=-1), dtype=tf.float32)
    hd = tf.abs((haversine_distance_tf(y_true, y_pred)))  # Shape = (batch_size, trajectory_length)
    mae = tf.math.reduce_sum(hd * mask, axis=1) / tf.math.reduce_sum(mask, axis=1)
    return mae
