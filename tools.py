def find_gpu_with_min_usage():
    import tensorflow as tf
    import subprocess
    import re
    process = subprocess.Popen(["gpustat"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()  # 获取输出
    memory_usage = re.findall(r'\|\s+(\d+)\s+/ 24564 MB', stdout)
    memory_usage = [int(x) for x in memory_usage]
    idx = memory_usage.index(min(memory_usage))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[idx], 'GPU')


def md5_sum(input):
    import hashlib

    # 要计算 MD5 的字符串
    return hashlib.md5(input.tobytes()).hexdigest()
