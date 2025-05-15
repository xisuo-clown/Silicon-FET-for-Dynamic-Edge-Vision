import os
from collections import defaultdict, Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.ma.core import indices
from spikingjelly.activation_based.model.tv_ref_classify.utils import accuracy


def plot_box(data, title):
    # 设置图形
    plt.rcParams.update(plt.rcParamsDefault)
    plt.figure(figsize=(10, 6))

    classes = len(data)  # 类别数
    samples_per_class = len(data[0])  # 每类样本数
    overall_mean = 0
    fontsize=15
    # 为每一类绘制点状图
    for i in range(classes):
        # 转换当前类的数据为浮点数
        class_data = list(map(float, data[i]))

        # 绘制每个类的散点图
        plt.scatter(np.ones(samples_per_class) * i, class_data, label=f'Class {i + 1}', alpha=0.6, s=50)

        # 计算并显示均值
        class_mean = np.mean(class_data)
        overall_mean += class_mean
        # 设置显示均值的位置，避免与数据点重叠
        plt.text(i, 0.93, f'{class_mean:.2f}', ha='center', va='bottom', fontsize=fontsize, color='black')
    overall_mean /= classes
    data = np.array(data, dtype=float)
    plt.text(5, np.min(data) + 0.01, f'overall accuracy: {overall_mean:.5f}', ha='center', va='bottom', fontsize=fontsize,
             color='black')
    plt.text(5, np.min(data) + 0.03, f'min accuracy: {np.min(np.mean(data, axis=0)):.5f}', ha='center', va='bottom',
             fontsize=fontsize, color='black')
    plt.text(5, np.min(data) + 0.05, f'max accuracy: {np.max(np.mean(data, axis=0)):.5f}', ha='center', va='bottom',
             fontsize=fontsize, color='black')
    plt.text(5, np.min(data) + 0.07, f'standard deviation: {np.std(np.mean(data, axis=0)):.5f}', ha='center', va='bottom',
             fontsize=fontsize, color='black')
    # 添加标签和标题
    plt.title('{}'.format(title))
    plt.xlabel('Class',fontsize=fontsize)
    plt.ylabel('Accuracy',fontsize=fontsize)
    plt.xticks(np.arange(classes), [f'{i + 1}' for i in range(classes)])
    plt.tick_params(axis='both', labelsize=fontsize)
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    # 显示图形
    plt.tight_layout()
    plt.show()


def csv_load(suffix: str, mode):
    import numpy as np
    from params_adjustment import find_path
    if mode:
        results_path = "test_results"
    else:
        results_path = "test_results_no_normalization"
    data = [[] for _ in range(10)]
    root_dir, train_save_path, test_save_path = find_path(suffix)
    results_path = os.path.join(test_save_path, results_path)
    csv_file = [f for f in os.listdir(results_path) if ".csv" in f]
    for csv in csv_file:
        with open(os.path.join(results_path, csv)) as f:
            df = f.readlines()
        for i in range(len(df)):
            df[i] = df[i].strip().split(',')
        for i in range(10):
            data[i].append(df[i][i])
    return data


def cal_for_brightness(suffix: str):
    x, y = load_npy_data(suffix)
    brightness_mean = np.mean(x, axis=(1, 2, 3))
    x_mean = [[] for _ in range(10)]
    x_g = classification_for_label(brightness_mean, x, y)
    for i in range(10):
        x_mean[i] = np.mean(x_g[i])
    x_overall = np.mean(np.array(x_g))
    return x_mean, x_overall


def classification_for_label(cal_results, x, y):
    x_g = [[] for _ in range(10)]
    for i in range(len(x)):
        x_g[int(y[i])].append(cal_results[i])
    return x_g


def load_npy_data(suffix):
    from params_adjustment import find_path
    root_dir, train_save_path, test_save_path = find_path(suffix)
    return np.load(os.path.join(train_save_path, "Aug_dataset_features.npy")), np.load(
        os.path.join(train_save_path, "Aug_dataset_labels.npy"))


def cal_for_contrast(suffix: str):
    x, y = load_npy_data(suffix)
    contrast_mean = np.std(x, axis=(1, 2)) / np.mean(x, axis=(1, 2))
    x_mean = [[] for _ in range(10)]
    x_g = classification_for_label(contrast_mean, x, y)
    for i in range(10):
        x_mean[i] = np.mean(x_g[i])
    x_overall = np.mean(np.array(x_g))
    return x_mean, x_overall


def cal_for_std(suffix: str):
    x, y = load_npy_data(suffix)
    contrast_mean = np.std(x, axis=(1, 2))
    x_mean = [[] for _ in range(10)]
    x_g = classification_for_label(contrast_mean, x, y)
    for i in range(10):
        x_mean[i] = np.mean(x_g[i])
    x_overall = np.mean(np.array(x_g))
    return x_mean, x_overall


def create_csv(data):
    import csv
    # 示例列表（二维）

    # 将列表写入 CSV 文件
    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("CSV file has been written.")


def load_one_picture(suffix: str, n=10):
    from params_adjustment import find_path
    root_dir, train_save_path, test_save_path = find_path(suffix)
    frame_path = os.path.join(train_save_path, f"{n}.npy")
    output_arr = np.load(frame_path)
    tags = ['Positive', 'Negative']
    cmap = 'viridis'
    fig, ax = plt.subplots(1, 2)
    plt.suptitle("number {} picture".format(n))
    ax[0].imshow(output_arr[:, :, 0], cmap=cmap)
    ax[0].set_title(tags[0])
    ax[1].imshow(output_arr[:, :, 1], cmap=cmap)
    ax[1].set_title(tags[1])
    brightness = np.mean(output_arr, axis=(0, 1))
    contrast = np.std(output_arr, axis=(0, 1)) / brightness
    fig.text(0.05, 0.9, f"Brightness: {brightness[0]:.2f}", fontsize=12, color='black', ha='left', va='top')
    fig.text(0.55, 0.9, f"Brightness: {brightness[1]:.2f}", fontsize=12, color='black', ha='left', va='top')
    fig.text(0.05, 0.85, f"Contrast: {contrast[0]:.2f}", fontsize=12, color='black', ha='left', va='top')
    fig.text(0.55, 0.85, f"Contrast: {contrast[1]:.2f}", fontsize=12, color='black', ha='left', va='top')
    fig.tight_layout()
    fig.show()


def csv_load_all(suffix: str, results_path: str):
    from params_adjustment import find_path

    data = [[] for _ in range(10)]
    root_dir, train_save_path, test_save_path = find_path(suffix)
    results_path = os.path.join(test_save_path, results_path)
    csv_file = [f for f in os.listdir(results_path) if ".csv" in f]
    for csv in csv_file:
        with open(os.path.join(results_path, csv)) as f:
            df = f.readlines()
        for i in range(len(df)):
            df[i] = df[i].strip().split(',')
        for i in range(10):
            data[i].append(df[i])
    return data


def plot_confusion_matrix_mean(data, title: str):
    import seaborn as sns
    data = np.array(data, dtype=float)
    data_mean = np.mean(data, axis=(1,))
    data_std = np.std(data, axis=(1,))
    matrix = [data_mean, data_std]
    matrix = np.round(matrix, 2)
    subtitle = ["mean", "std"]
    for i in range(2):
        axis_labels = list(range(1, matrix[i].shape[0] + 1))

        # 画热力图
        plt.figure(figsize=(8, 8))
        TITLE_FONT_SIZE = {"size": "22"}
        LABEL_SIZE = 20
        sns.set(font_scale=1.5)
        g = sns.heatmap(matrix[i], annot=True, fmt='g', cbar=False, cmap='Blues')

        # 设置刻度
        g.set_xticks(range(matrix[i].shape[1]))
        g.set_yticks(range(matrix[i].shape[0]))
        g.set_xticklabels(axis_labels)
        g.set_yticklabels(axis_labels)
        g.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)

        # 旋转刻度标签
        plt.setp(g.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        plt.setp(g.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # 轴标签和标题
        plt.xlabel("Predicted", fontdict=TITLE_FONT_SIZE, fontweight='bold')
        plt.ylabel("Actual", fontdict=TITLE_FONT_SIZE, fontweight='bold')
        plt.title("{} {}".format(title, subtitle[i]), fontdict=TITLE_FONT_SIZE, fontweight='bold')

        plt.show()


def load_and_test(suffix: str):
    from params_adjustment import find_path
    root_dir, train_save_path, test_save_path = find_path(suffix)
    results_path = os.path.join(test_save_path, "", "test_results_without_aug")

    x_test = np.load(os.path.join(test_save_path, "dataset_features.npy"))
    y_test = np.array(np.load(os.path.join(test_save_path, "dataset_labels.npy")), dtype=int)
    l = [0, 8, 9]
    idx_store = defaultdict(list)

    for f in os.listdir(results_path):
        if f.endswith(".h5"):
            model = load_model(os.path.join(results_path, f))
            accuracy_file, accuracy_model = model_verification(model, x_test, y_test, f, results_path)
            print(f.replace(".h5", ""))
            print("accuracy_file:", accuracy_file)
            print("accuracy_model:", accuracy_model)
            try:
                if accuracy_file == accuracy_model:
                    print("model verification results:", accuracy_file == accuracy_model)
                else:
                    raise ValueError("An error occurred: Check Error")
            except ValueError:
                with open("error_h5.txt", 'a+') as s:
                    s.writelines(os.path.join(results_path, f))
                    continue

            y_pred = model.predict(x_test)
            y_pred = np.array(np.argmax(y_pred, axis=1), dtype=int)
            for i in l:
                for j in range(len(y_pred)):
                    if y_pred[j] != i and y_test[j] == i and y_pred[j] in l:
                        idx_store[f"actual:{y_test[j]} predict:{y_pred[j]}"].append(j)
    idx_store_max = defaultdict(tuple)
    for k, v in idx_store.items():
        counter = Counter(idx_store[k])
        key = max(counter, key=counter.get)
        length = counter[key]
        idx_store_max[k] = (
            "idx:{}".format(key), "misclassified times:{}".format(length))
    return idx_store, idx_store_max


def model_verification(model, x_test, y_test, model_path, results_path):
    accuracy_model = str(model.evaluate(x_test, y_test)[1])
    accuracy_file = 0
    txt_file = model_path.replace(".h5", ".txt")
    txt_file = os.path.join(results_path, txt_file)
    with open(txt_file, 'r') as file:
        for line in file:
            # 检查该行是否包含 'accuracy'
            if 'accuracy' in line:
                # 提取 'accuracy' 后面的数值
                accuracy_file = line.split('accuracy:')[1].strip().replace(',', '')
    return accuracy_file, accuracy_model


def load_model(model_path):
    from params_adjustment import ResNet18
    import tensorflow as tf
    model = ResNet18(10)
    from tensorflow.keras.layers import Input
    model.build(input_shape=(None, 128, 128, 2))
    model.build_graph()

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        # initial_learning_rate,
        initial_learning_rate=0.01,
        decay_steps=3000,
        decay_rate=0.5,
        staircase=True)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        # optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"])
    model.load_weights(model_path)
    # model.summary()
    return model


def grad_cam(model, img_array, class_index, if_aug: bool):
    """
    计算 ResNet18 的 Grad-CAM 热力图
    :param model: 训练好的 ResNet18 模型
    :param img_array: 预处理后的输入图像 (batch_size, height, width, channels)
    :param class_index: 目标类别索引
    :param layer_name: 目标卷积层
    :return: 归一化的热力图
    """
    import tensorflow as tf
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

    # 创建一个 tf.keras.Input 层，形状与输入图像相匹配
    # model=model.build_graph()
    # model.summary()
    from tensorflow.keras.layers import Input
    if if_aug:
        inputs = model.get_layer("conv2d").input
        outputs = [model.get_layer("resnet_block_2").output, model.get_layer("dense").output]
    else:
        inputs = model.get_layer("conv2d_20").input
        outputs = [model.get_layer("resnet_block_10").output, model.get_layer("dense_1").output]
    grad_model = tf.keras.models.Model(
        inputs=inputs,  # 使用原模型的输入
        outputs=outputs  # 输出目标层和最终模型输出
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        loss = predictions[:, class_index]  # 目标类别的概率

    # 计算梯度
    grads = tape.gradient(loss, conv_output)[0]

    # 计算通道权重（全局平均池化）
    weights = tf.reduce_mean(grads, axis=(0, 1))

    # 计算加权特征图
    cam = tf.reduce_sum(tf.multiply(weights, conv_output[0]), axis=-1)

    # 归一化到 [0,1]
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    return cam


def overlay_heatmap(heatmap, img, alpha=0.4):
    """
    叠加热力图到原始图像上（Matplotlib 版本）
    :param heatmap: 归一化的热力图
    :param img: 原始图像 (H, W, C)，已归一化到 [0,1]
    :param alpha: 叠加透明度
    :return: 叠加后的图像
    """
    heatmap = np.uint8(255 * heatmap)  # 变为 0-255
    heatmap = plt.cm.jet(heatmap)[:, :, :3]  # 使用 Jet 颜色映射（去掉 alpha 通道）
    heatmap = np.uint8(255 * heatmap)  # 变为 0-255
    heatmap = heatmap / 255.0  # 归一化

    # 叠加到原图
    superimposed_img = alpha * heatmap + (1 - alpha) * img
    superimposed_img = np.clip(superimposed_img, 0, 1)  # 限制范围 0-1

    return superimposed_img


def show_pic(suffix, idx, title: str, if_train: bool = False):
    from params_adjustment import find_path
    root_dir, train_save_path, test_save_path = find_path(suffix)
    results_path = os.path.join(test_save_path, "archieve", "test_results_no_normalization")

    x_test = np.load(os.path.join(test_save_path, "dataset_features.npy"))
    y_test = np.load(os.path.join(test_save_path, "dataset_labels.npy"))
    output_arr = x_test[idx]
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    tags = ['Positive', 'Negative']
    # cmap = ['OrRd', 'BuPu']
    # cmap = 'magma'
    cmap = 'viridis'
    fig, ax = plt.subplots(1, 2)
    # plt.suptitle("Test set {0}, Class {1}".format(a, label[n]))
    plt.suptitle(title)
    ax[0].imshow(output_arr[:, :, 0], cmap=cmap, vmin=0, vmax=20)
    ax[0].set_title(tags[0])
    ax[1].imshow(output_arr[:, :, 1], cmap=cmap, vmin=0, vmax=20)
    ax[1].set_title(tags[1])
    ax[0].grid(False)
    ax[1].grid(False)
    fig.tight_layout()
    # fig.colorbar
    # print(output_arr[:, :, 0]*1E5)
    fig.show()


def show_his(suffix, idx, title: str, if_train: bool = False):
    from params_adjustment import find_path
    root_dir, train_save_path, test_save_path = find_path(suffix)
    results_path = os.path.join(test_save_path, "archieve", "test_results_no_normalization")

    x_test = np.load(os.path.join(test_save_path, "dataset_features.npy"))
    y_test = np.load(os.path.join(test_save_path, "dataset_labels.npy"))
    output_arr = x_test[idx]
    # 绘制直方图
    plt.figure(figsize=(8, 6))
    plt.hist(output_arr.flatten(), bins=50, color='blue', alpha=0.7)
    plt.title(f"{title} - Histogram of output_arr (Index {idx})")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


def show_pic_by_content(output_arr):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    tags = ['Positive', 'Negative']
    # cmap = ['OrRd', 'BuPu']
    # cmap = 'magma'
    cmap = 'viridis'
    fig, ax = plt.subplots(1, 1)
    # plt.suptitle("Test set {0}, Class {1}".format(a, label[n]))
    plt.suptitle("brightness:{}".format(np.mean(output_arr)))
    ax.imshow(output_arr[:, :], cmap=cmap, vmin=0, vmax=25)
    ax.set_title(tags[0])
    fig.tight_layout()
    # fig.colorbar
    # print(output_arr[:, :, 0]*1E5)
    fig.show()


def show_pic_by_number(suffix: str, n, if_train: bool = False, if_aug: bool = False):
    from params_adjustment import find_path
    root_dir, train_save_path, test_save_path = find_path(suffix)
    if not if_aug:
        pic_path = os.path.join(train_save_path, f"{n}.npy")
    else:
        pic_path = os.path.join(train_save_path, f"Aug_{n}.npy")
    arr = np.load(pic_path)
    from new_try_for_data_gen import aug_process
    # from params_adjustment import aug_process
    arr = np.expand_dims(arr, axis=0)
    for output_arr in arr:
        show_pic_by_content(output_arr[:, :, 0])
    return arr


def gen_frame(x, y, t, i_d):
    c = 3E-4
    n_clip = 3
    t = [(t0 - t[0]) * 1e-6 * c for t0 in t]
    from new_try_for_data_gen import index_arr
    indexarr = index_arr()
    from collections import defaultdict
    events_dict_pos = defaultdict(list)
    events_dict_neg = defaultdict(list)

    events_dict_pos_time = defaultdict(list)
    events_dict_neg_time = defaultdict(list)

    dict_pos = {}
    dict_neg = {}
    end_time = []

    dict_neg_time = {}
    dict_pos_time = {}
    # j range(n): split event stream within ne parts,
    # Each part contains n event in whole event stream

    t_whole = 6 * c
    t_clip = t_whole / n_clip

    n = 0

    ne = len(t) - 1
    # print('n is {}'.format(n))
    # print('ne is {}'.format(ne))
    end_t = (t[ne] - t[n])
    for h in range(128 * 128):
        events_dict_pos[h] = []
        events_dict_neg[h] = []
        events_dict_pos_time[h] = []
        events_dict_neg_time[h] = []
        # print(len(t))
    for i in range(len(t)):
        key = indexarr[x[i]][y[i]]
        if ne > i > n:
            events_dict_pos[key].append((t[i] - t[n]))

    end_time.append(end_t)

    dict_pos[0] = dict(events_dict_pos)
    dict_neg[0] = dict(events_dict_neg)
    # here we want to generate a dictionary, contains the interval time between two events
    list_dict = [events_dict_neg, events_dict_pos]
    list_dict_time = [events_dict_neg_time, events_dict_pos_time]
    for m in range(2):
        events_dict = list_dict[m]
        events_dict_time = list_dict_time[m]
        for k in range(128 * 128):
            if events_dict[k]:
                # for l in events_dict[k]:
                for l in events_dict[k]:
                    if l != events_dict[k][0]:
                        events_dict_time[k].append(l - l_0)
                    l_0 = l
                events_dict_time[k].append(end_t - events_dict[k][-1])
    dict_pos_time[0] = dict(events_dict_pos_time)
    dict_neg_time[0] = dict(events_dict_neg_time)
    events_dict = dict_pos_time[0]
    temp_save = []
    for i in range(128 * 128):
        if events_dict[i]:
            id_last = i_d.d[0]
            for j in events_dict[i]:

                y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_, l_a, l_b = i_d.get_para(id_last)
                from params_adjustment import id_time_new
                id_b, id_a = id_time_new(id_last, j, y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_, l_a, l_b)

                if j != events_dict[i][-1]:
                    id_last = id_a

                else:
                    id_last = id_b

            if id_last - 15.2 > 0:
                temp_save.append(id_last - 15.2)

            else:
                temp_save.append(0)
            # # >>>>>>>>>>??????????????????????
            # if id_last - 15.2 < 0:
            #     print(id_last)

            # temp_save[polar].append(id_last)
        else:
            temp_save.append(0)
    output_arr = np.empty((128, 128))
    for k in range(128):
        for m in range(128):
            output_arr[k, m] = temp_save[int(indexarr[m][k])]
    return output_arr


def get_accuracy_for_csv(t_no_normalization):
    t_data = [[] for _ in range(10)]
    for i in range(len(t_no_normalization)):
        for j in range(len(t_no_normalization[i])):
            t_data[i].append(t_no_normalization[i][j][i])
    return t_data


def create_dict(x, y):
    dic = defaultdict(list)
    for i in range(len(x)):
        dic[y[i]].append(x[i])
    return dic


def cal_brightness_overall(x, y, if_by_size: bool = True):
    dic = create_dict(x, y)
    dic_brightness = defaultdict(list)
    if if_by_size:
        for k, v in dic.items():
            dic_brightness[k] = np.mean(v)

    else:
        for k, v in dic.items():
            dic_brightness[k] = np.mean(np.sum(v, axis=(1, 2)) / np.count_nonzero(v, axis=(1, 2)))
    return dic_brightness


def get_brightness_distribution(x, y):
    dic = create_dict(x, y)
    dic_brightness = defaultdict(list)
    for k, v in dic.items():
        dic_brightness[k] = np.mean(v, axis=(1, 2, 3))
    return dic_brightness


def cal_std(x, y, if_by_size: bool = True):
    dic = create_dict(x, y)
    dic_brightness = defaultdict(list)
    if if_by_size:
        for k, v in dic.items():
            dic_brightness[k] = np.std(v)

    else:
        for k, v in dic.items():
            dic_brightness[k] = np.std(np.sum(v, axis=(1, 2)) / np.count_nonzero(v, axis=(1, 2)))
    return dic_brightness


def load_dataset(suffix: str, if_train: bool = True):
    from params_adjustment import find_path
    root_dir, train_save_path, test_save_path = find_path(suffix)
    if if_train:
        return np.load(os.path.join(train_save_path, "Aug_dataset_features.npy")), np.load(
            os.path.join(train_save_path, "Aug_dataset_labels.npy"))
    return np.load(os.path.join(test_save_path, "dataset_features.npy")), np.load(
        os.path.join(test_save_path, "dataset_labels.npy"))


def grad_cam_test():
    global root_dir, train_save_path, test_save_path, indices, model_path_without_aug, model_path_aug, model_aug, model_without_aug, i
    from params_adjustment import find_path
    import scipy.ndimage
    # br=cal_for_std("no_tune_6clips")
    root_dir, train_save_path, test_save_path = find_path("no_tune_6clips")
    y = np.load(os.path.join(test_save_path, "dataset_labels.npy"))
    x_ori = np.load(os.path.join(test_save_path, "dataset_features.npy"))
    x = x_ori[y == 8]
    indices = np.where(y == 8)[0]
    model_path_without_aug = os.path.join(test_save_path, "test_results_without_aug", "20250227_171348_0.h5")
    model_path_aug = os.path.join(test_save_path, "test_results_with_aug", "20250227_190025_0.h5")
    # model_aug = load_model(model_path_aug)
    model_without_aug = load_model(model_path_without_aug)
    for i in range(2):
        for j in range(len(x)):
            aug = bool(i)
            img = x[j]
            if aug:
                model = model_aug
                img *= 1.2
            else:
                model = model_without_aug
            img = np.expand_dims(img, axis=0)
            img[0, :, :, 0] = scipy.ndimage.median_filter(img[0, :, :, 0], size=3)
            img[0, :, :, 1] = scipy.ndimage.median_filter(img[0, :, :, 1], size=3)
            # 计算 Grad-CAM 热力图
            class_index = np.argmax(model.predict(img))
            # class_index = 8
            print(class_index)

            heatmap = grad_cam(model, img, class_index, if_aug=aug)

            # 叠加热力图
            # superimposed_img = overlay_heatmap(heatmap, img)
            superimposed_img = heatmap
            # 显示结果
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(img[0, :, :, 0], vmin=0, vmax=25)
            plt.title("Original Image")
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.imshow(superimposed_img)
            plt.title("Grad-CAM Heatmap")
            plt.axis("off")

            plt.suptitle(
                "Grad-CAM Visualization for {} idx:{}\n predicted:{}".format("with aug" if aug else "without aug",
                                                                             indices[j], class_index),
                fontsize=16, fontweight="bold")

            plt.show()


def plot_box_test(suffix, result_path, tt):
    t_no_normalization = csv_load_all(suffix, result_path)
    t_data = get_accuracy_for_csv(t_no_normalization)
    plot_box(t_data, "Accuracy Distribution")
    plot_confusion_matrix_mean(t_no_normalization, "Confusion Matrix")

def draw_comparative_chart_for_suffix(suffix1,suffix2,path1,path2):
    t_data=[0,0]
    t_no_normalization= csv_load_all(suffix1, path1)
    t_data[0] = get_accuracy_for_csv(t_no_normalization)
    t_no_normalization = csv_load_all(suffix2, path2)
    t_data[1] = get_accuracy_for_csv(t_no_normalization)
    t_data[0],t_data[1] = np.array(t_data[0],dtype=float),np.array(t_data[1],dtype=float)
    t_data[0],t_data[1]=np.mean(t_data[0],axis=1),np.mean(t_data[1],axis=1)
    draw_comparative_chart(t_data[0],t_data[1])

def draw_comparative_chart(accuracy_before,accuracy_after):
    import numpy as np
    import matplotlib.pyplot as plt

    # 10 个类别的名称
    categories = [f'Class {i + 1}' for i in range(10)]

    # 示例准确率数据（请替换为你的实际数据）


    # 计算 Overall Accuracy
    overall_before = np.mean(accuracy_before)
    overall_after = np.mean(accuracy_after)

    # 设置柱状图的位置
    x = np.arange(len(categories))  # 类别的索引
    width = 0.4  # 增加柱子的宽度

    # 创建柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_before = ax.bar(x - width / 2, accuracy_before, width, label='ResNet', color='blue', alpha=0.7)
    bars_after = ax.bar(x + width / 2, accuracy_after, width, label='ResNet-LSTM', color='green', alpha=0.7)

    # 在柱子上方添加准确率标注
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f'{height:.2}',
                    ha='center', va='bottom', fontsize=10, color='black')

    add_labels(bars_before)
    add_labels(bars_after)

    # 添加 Overall Accuracy 文字标注
    ax.axhline(overall_before, color='blue', linestyle='--', linewidth=2, label=f'Overall ResNet: {overall_before:.2}')
    ax.axhline(overall_after, color='green', linestyle='--', linewidth=2,
               label=f'Overall ResNet-LSTM: {overall_after:.2}')

    # 设置轴标签和标题
    ax.set_xlabel('Categories', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Comparison of Classification Accuracy', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45)  # 旋转类别标签以提高可读性
    ax.set_ylim(0, 1.1)  # 设定 y 轴范围，并给顶部留出空间放标注

    # 添加图例
    ax.legend(loc='upper center', bbox_to_anchor=(0.35, 0.15), ncol=2)

    # 显示图表
    plt.show()


if __name__ == '__main__':
    # x,y=load_dataset("no_tune_6clips",if_train=False)
    # brightness_dis=get_brightness_distribution(x,y)
    # from gaussian_distribution import plot_gaussian_distribution
    #
    # plot_gaussian_distribution(brightness_dis[0])
    # plot_gaussian_distribution(brightness_dis[8])

    # grad_cam_test()

    # x, y = load_npy_data("no_tune_6clips")
    # bri = np.mean(x, axis=(1, 2, 3))
    # x_ = classification_for_label(bri, x, y)
    # from gaussian_distribution import plot_distribution
    #
    # data = [x_[0], x_[8]]
    # label = ["class 1", "class 9"]
    # plot_distribution(data, label, "brightness")
    # from params_adjustment import check_for_files, find_path
    #
    # check_for_files("no_tune", "tune_for_all")
    # check_for_files("no_tune", "tune_for_eight")
    # check_for_files("no_tune", "tune_for_nine")
    # for i in range(0,3600):
    #     try:
    #         show_pic_by_number("tune_for_all_new_10", i, True,False)
    #     except Exception as e:
    #         print(e)
    draw_comparative_chart_for_suffix("no_tune_6clips","ltsm_resnet_6clips","test_results_without_aug","test_results_3_resnet")
    tune_type = ["no_tune_6clips", "tune_for_all_new", "tune_for_eight_new"]
    title = ["no tune", "tune for all", "tune for eight"]
    plot_box_test("lstm_5clips","test_results_3_resnet_128_hidden_units", "3 clips")
    # plot accuracy distribution
    # suffix = "ltsm_resnet_6clips"
    # result_path = "test_results"
    # tt = "with 6 resnet blocks"
    # plot_box_test(suffix,result_path, tt)





    suffix = [
        "no_tune_6clips",
        "no_tune_6clips",
        "ltsm_resnet_6clips",
        "ltsm_resnet_6clips",
        "ltsm_resnet_6clips"
    ]

    result_path = [
        "test_results_without_aug",
        "resnet_num_6",
        "test_results",
        "test_results_3_resnet",
        "test_results_3_resnet_64_hidden_units"
    ]

    tt = [
        "with 3 resnet blocks",
        "with 6 resnet blocks",
        "with 6 resnet blocks",
        "with 3 resnet blocks",
        "with 3 resnet blocks"
    ]

    for i in range(len(tt)):


        plot_box_test(suffix[i],result_path[i], tt[i])



    for i in range(len(tune_type)-2):
        print("round of ", i, " for ", title[i])
        idx_store, idx_store_max = load_and_test(tune_type[i])
        for k, v in idx_store_max.items():
            idx = int(v[0].split("idx:")[1])
            title_fin = title[i] + "\n" + k + "\n" + "\n" + v[0] + "\n" + v[1]
            show_pic(tune_type[i], idx, title_fin)
            # show_his(tune_type[i], idx, title_fin)
        print(idx_store)
        print(idx_store_max)
    # brightness_mean = [0, 0]
    # brightness_overall = [0, 0]
    # contrast_mean = [0, 0]
    # contrast_overall = [0, 0]
    # std_mean = [0, 0]
    # std_overall = [0, 0]
    # std_mean[0], std_overall[0] = cal_for_std("no_tune")
    # std_mean[1], std_overall[1] = cal_for_std("tune_for_all")
    # # brightness_mean[0],brightness_overall[0]=cal_for_brightness("no_tune")
    # # brightness_mean[1],brightness_overall[1]=cal_for_brightness("tune_for_all")
    # # contrast_mean[0],contrast_overall[0]=cal_for_contrast("no_tune")
    # # contrast_mean[1],contrast_overall[1]=cal_for_contrast("tune_for_all")
    # create_csv(c1)
    # create_csv(c2)
    # print()
