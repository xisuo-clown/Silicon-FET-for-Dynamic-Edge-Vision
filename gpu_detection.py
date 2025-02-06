import tensorflow as tf


def gpu_list():
    devices = tf.config.list_physical_devices()
    # 检查是否有GPU
    has_gpu = any(tf.config.list_physical_devices('GPU'))
    print("Is GPU available:", has_gpu)


# 列出所有可用的物理设备
