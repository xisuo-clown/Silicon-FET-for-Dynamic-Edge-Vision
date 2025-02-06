import time

import numpy as np


########################################################@@@@display fig(gesture, with polarity)
def img_show(if_train: bool, a: int,if_tune:bool=False):
    import os
    import event_stream
    root_dir, train_save_path, test_save_path = event_stream.find_path()
    from event_stream import n_num
    if if_train:
        dataset_name = 'Train'
        dataset_path = train_save_path
        path_1 = dataset_path + "dataset_labels_augmentation.npy"
        path_2 = dataset_path + 'dataset_features_augmentation.npy'
        path_3 = dataset_path + "Aug_dataset_labels_remove.npy"
        path_4 = dataset_path + 'Aug_dataset_features_remove.npy'
    else:
        dataset_name = 'Test'
        dataset_path = test_save_path
        path_1 = dataset_path + "dataset_labels.npy"
        path_2 = dataset_path + 'dataset_features.npy'
        path_3 = dataset_path + "dataset_labels_remove.npy"
        path_4 = dataset_path + 'dataset_features_remove.npy'
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    tags = ['Positive', 'Negative']
    # cmap = ['OrRd', 'BuPu']
    # cmap = 'magma'
    cmap = 'viridis'

    # n = int(n_num * a + (n_num - 1))
    # print(n, n_num)
    ###################show a given frame, before stack
    label_list = np.load(dataset_path + "dataset_labels.npy", allow_pickle=True)
    n=a*n_num-1
    label = label_list[n]

    if if_tune:
        output_arr = np.load(dataset_path + 'class8_tuned_{0}.npy'.format(n), allow_pickle=True)
    else:
        output_arr = np.load(dataset_path + '{0}.npy'.format(n), allow_pickle=True)
    # output_arr_stack = np.load(dataset_path + 'dataset_features_remove.npy', allow_pickle=True)
    # print(output_arr[0].shape)
    # output_arr = output_arr[0]
    print(output_arr.shape)
    # ########show a given frame, after stack, and augmentation
    # output_arr_stack = np.load(path_2, allow_pickle=True)
    # output_arr = output_arr_stack[n]
    # print(output_arr.shape)
    # label_list = np.load(path_1, allow_pickle=True)
    # label = label_list[n]
    # # #################################
    # ########show a given frame, after stack, remove the extra set,and augmentation
    # output_arr_stack = np.load(path_4, allow_pickle=True)
    # output_arr = output_arr_stack[n]
    # print(output_arr.shape)
    # label_list = np.load(path_3, allow_pickle=True)
    # label = label_list[n]
    # ######################display negative and positive###############
    fig, ax = plt.subplots(1, 2)
    # plt.suptitle("Test set {0}, Class {1}".format(a, label[n]))
    plt.suptitle("{0} set {1}, Class {2}".format(dataset_name, a, label))
    ax[0].imshow(output_arr[:, :, 0], cmap=cmap)
    ax[0].set_title(tags[0])
    ax[1].imshow(output_arr[:, :, 1], cmap=cmap)
    ax[1].set_title(tags[1])
    fig.tight_layout()
    # fig.colorbar
    # print(output_arr[:, :, 0]*1E5)
    fig.show()
    # ##########dispaly neg/positive end##############
    # import matplotlib.pyplot as plt
    plt.imshow(output_arr[:, :, 1], cmap=cmap)
    plt.colorbar(plt.imshow(output_arr[:, :, 1]), cmap=cmap)
    plt.show()


def img_show_tuning(if_train, n, class_num):
    import os
    import event_stream
    root_dir, train_save_path, test_save_path = event_stream.find_path()
    from event_stream import n_num
    if if_train:
        dataset_name = 'Train'
        dataset_path = train_save_path

    else:
        dataset_name = 'Test'
        dataset_path = test_save_path


    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    tags = ['Positive', 'Negative']
    # cmap = ['OrRd', 'BuPu']
    # cmap = 'magma'
    cmap = 'viridis'
    ###################show a given frame, after tuning
    try:
        index_arr = np.load(dataset_path + 'class{0}_tuned_dataset_index.npy'.format(
            class_num), allow_pickle=True)
        print("Index Exist")
        index_list = index_arr.tolist()
        output_arr = np.load(dataset_path + 'class{0}_tuned_{1}.npy'.format(
            class_num, index_list[n]), allow_pickle=True)
        print(output_arr.shape)
    except:
        print("Fail to find index")
    # ######################display negative and positive###############
    fig, ax = plt.subplots(1, 2)
    # plt.suptitle("Test set {0}, Class {1}".format(a, label[n]))
    if class_num > 2:
        class_num += 1
        serial_numbers = int(index_list[n] / 5) + 98


    max_value = np.max(output_arr)
    min_value = np.min(output_arr)
    print(max_value)
    output_arr = output_arr / (max_value-min_value)


    plt.suptitle("After tuning: {0} set: {1}, Class {2}".format(dataset_name, serial_numbers, class_num))
    ax[0].imshow(output_arr[:, :, 0], cmap=cmap)
    ax[0].set_title(tags[0])
    ax[1].imshow(output_arr[:, :, 1], cmap=cmap)
    ax[1].set_title(tags[1])
    fig.tight_layout()
    # fig.colorbar
    # print(output_arr[:, :, 0]*1E5)
    fig.show()
    # ##########dispaly neg/positive end##############
    # import matplotlib.pyplot as plt
    plt.imshow(output_arr[:, :, 1], cmap=cmap)
    plt.colorbar(plt.imshow(output_arr[:, :, 1]), cmap=cmap)
    plt.show()

def img_show_before_tuning(if_train, n, class_num):

    import os
    import event_stream
    root_dir, train_save_path, test_save_path = event_stream.find_path()
    from event_stream import n_num
    if if_train:
        dataset_name = 'Train'
        dataset_path = train_save_path
        path_1 = dataset_path + "dataset_labels_augmentation.npy"
        path_2 = dataset_path + 'dataset_features_augmentation.npy'
        path_3 = dataset_path + "Aug_dataset_labels_remove.npy"
        path_4 = dataset_path + 'Aug_dataset_features_remove.npy'
    else:
        dataset_name = 'Test'
        dataset_path = test_save_path
        path_1 = dataset_path + "dataset_labels.npy"
        path_2 = dataset_path + 'dataset_features.npy'
        path_3 = dataset_path + "dataset_labels_remove.npy"
        path_4 = dataset_path + 'dataset_features_remove.npy'
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    tags = ['Positive', 'Negative']
    # cmap = ['OrRd', 'BuPu']
    # cmap = 'magma'
    cmap = 'viridis'

    label_list = np.load(path_3, allow_pickle=True)

    for i in range(len(label_list)):
        if class_num == label_list[i]:
            n_real = i
            break


    ###################show a given frame, before stack
    output_arr = np.load(path_4, allow_pickle=True)
    # output_arr_stack = np.load(dataset_path + 'dataset_features_remove.npy', allow_pickle=True)
    # print(output_arr[0].shape)
    # output_arr = output_arr[0]

    max_value = np.max(output_arr)
    min_value = np.min(output_arr)
    output_arr = output_arr / (max_value-min_value)

    # ######################display negative and positive###############
    fig, ax = plt.subplots(1, 2)
    # plt.suptitle("Test set {0}, Class {1}".format(a, label[n]))
    plt.suptitle("Before tuning: {0} set {1}, Class {2}".format(dataset_name, n_real, class_num))
    ax[0].imshow(output_arr[n_real,:,:,0], cmap=cmap)
    ax[0].set_title(tags[0])
    ax[1].imshow(output_arr[n_real,:,:,1], cmap=cmap)
    ax[1].set_title(tags[1])
    fig.tight_layout()
    # fig.colorbar
    # print(output_arr[:, :, 0]*1E5)
    fig.show()
    # ##########dispaly neg/positive end##############
    # import matplotlib.pyplot as plt
    plt.imshow(output_arr[n_real, :, :, 1], cmap=cmap)
    plt.colorbar(plt.imshow(output_arr[n_real, :, :, 1]), cmap=cmap)
    plt.show()


# ###########################display the figure after augmentation
# here, we only display the positive side
def img_show_aug(if_train, n):
    import os
    from event_stream import find_path
    root_dir, train_save_path, test_save_path = find_path()
    path = train_save_path
    import matplotlib.pyplot as plt
    output_arr = np.load(os.path.join(path, '{0}.npy'.format(n)), allow_pickle=True)
    label = np.load(os.path.join(path, "dataset_labels.npy"), allow_pickle=True)
    tags = ['Original', 'Negative Rotation', 'Positive Rotation', 'Translation_Origin',
            'Translation_Neg_Rotate', 'Translation_Pos_Rotate']
    fig, ax = plt.subplots(2, 3)
    if if_train:
        plt.suptitle("Augmentation: Train set {0}, Class {1}".format(n, label[n]))
    else:
        plt.suptitle("Augmentation: Test set {0}, Class {1}".format(n, label[n]))
    x = rotate_shift_gen(output_arr)
    for j in range(2):
        for i in range(3):
            ax[j][i].imshow(x[int(3 * j + i), :, :, 0])
            ax[j][i].set_title(tags[3 * j + i])
    fig.tight_layout()
    fig.show()


def rotate_shift_gen(output_arr):
    import tensorflow as tf
    from numpy import expand_dims
    from tqdm import tqdm
    # rotate and shift
    x_rota1 = tf.keras.layers.RandomRotation(factor=(-0.1, 0), fill_mode='reflect')
    x_rota2 = tf.keras.layers.RandomRotation(factor=(0, 0.1), fill_mode='reflect')
    x_shift = tf.keras.layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1),
                                                fill_mode='reflect', fill_value=0.0)
    x_shift_1 = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=(-0.1, 0), fill_mode='constant'),
        tf.keras.layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1),
                                          fill_mode='constant', fill_value=0.0)])
    x_shift_2 = tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=(0, 0.1), fill_mode='constant'),
        tf.keras.layers.RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1),
                                          fill_mode='constant', fill_value=0.0)])
    x = output_arr
    output_arr = output_arr[0]
    x = np.row_stack((x, expand_dims(x_rota1(output_arr), axis=0)))
    x = np.row_stack((x, expand_dims(x_rota2(output_arr), axis=0)))
    x = np.row_stack((x, expand_dims(x_shift(output_arr), axis=0)))
    x = np.row_stack((x, expand_dims(x_shift_1(output_arr), axis=0)))
    x = np.row_stack((x, expand_dims(x_shift_2(output_arr), axis=0)))
    print(x.shape)
    return x


import matplotlib.pyplot as plt


# display the event flow
def visualize_eventflow(if_train, n):
    # 2 color display the positive/ negative
    # we generate a 2 s event stream and a full 6 seconds event stream
    plot_scatter(if_train, n, aspect=(3, 1, 1), i=1)
    plot_scatter(if_train, n, aspect=(2, 1, 1), i=3)


def gen_scatter_data(if_train, n, i):
    from event_stream import init_event
    train, test = init_event()
    if if_train:
        event_stream = train
    else:
        event_stream = test

    event, label = event_stream[n]
    x = tuple(event["x"])
    y = tuple(event["y"])
    t = tuple(event["t"])
    p = tuple(event["p"])

    x_pos = []
    y_pos = []
    t_pos = []
    x_neg = []
    y_neg = []
    t_neg = []
    for i in range(int(len(t) / i)):
        if (t[i] - t[0]) * 1E-6 < 2:
            if p[i]:
                x_pos.append(x[i])
                y_pos.append(y[i])
                t_pos.append((t[i] - t[0]) * 1E-6)
            else:
                x_neg.append(x[i])
                y_neg.append(y[i])
                t_neg.append((t[i] - t[0]) * 1E-6)

    return t_pos, x_pos, y_pos, t_neg, x_neg, y_neg, label


def plot_scatter(if_train, n, aspect, i):
    t_pos, x_pos, y_pos, t_neg, x_neg, y_neg, label = gen_scatter_data(if_train, n, i)
    # fig = plt.figure(figsize=(20, 15))
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(projection='3d')
    # ax.set_box_aspect((2, 1, 1))
    ax.set_box_aspect(aspect)
    cmp = ['autumn_r', 'winter_r']

    ax.scatter(t_pos, x_pos, y_pos, label='Positive', cmap=cmp[0], c=t_pos, s=2, alpha=0.2)
    ax.scatter(t_neg, x_neg, y_neg, label='Negative', cmap=cmp[1], c=t_neg, s=2, alpha=0.2)

    # ax.set_xticks()
    # ax.set_yticks([])
    # ax.set_zticks([])
    ax.tick_params(axis='x', labelsize=20)
    plt.title("simple 3D scatter plot, number {0}, class {1}".format(n, label))
    # ax.set_xlabel('Time', fontdict=dict(weight='bold', fontsize=20))
    plt.show()


# ##############dispaly 3 frames in same event stream, each of frame contains 30ms
# only display positive side
def display_3_frames():
    from spikingjelly.datasets import play_frame
    begin_frame = 80
    interval = 2

    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture as D128G
    start_time = time.time()
    root_dir = ('C:/Users/ASUS/OneDrive - Nanyang Technological University/datasets/DVS128Gesture/')
    # separate 6 s event into 200 frames, each last 30ms
    data_set = D128G(root_dir, train=True, data_type='frame', frames_number=100, split_by='number')
    data_num = 1176

    # frame, label = data_set[n]
    # print(frame.shape)
    cmap = 'magma_r'
    # 'pink_r'
    # 'Oranges'
    # 'magma'

    pad = 5  # in points

    fig, ax = plt.subplots(10, 4, figsize=(12, 20))
    listn = []
    for i in range(10):
        if i < 2:
            n = 10 + 98 * i
        elif 8 > i >= 2:
            n = 10 + 98 * (i + 1)
        else:
            n = 10 + 98 * (i + 2)
        frame, label = data_set[n]
        for j in range(4):
            ax[i][j].imshow(frame[begin_frame + j * interval, 0, :, :], cmap=cmap)
            # ax[i][j].axis("off")
            ax[i, j].xaxis.set_ticklabels([])
            ax[i, j].yaxis.set_ticklabels([])
        listn.append(label)

    # rows = ['Class {}'.format(row) for row in listn]
    t1 = ['Clapping\nhands', 'Right hand\nwave', 'Left hand\nwave',
          'Right arm\nclockwise', 'Right arm\ncounter\n-clockwise',
          'Left arm\nclockwise', 'Left arm\ncounter\n-clockwise',
          'Arm roll', 'Air drums', 'Air guitar']
    # rows = ['Class {}'.format(row) for row in range(10)]
    for ax, row in zip(ax[:, 0], t1):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    # size='large',
                    ha='right', va='center', weight='bold', fontsize=22)
    fig.tight_layout()
    fig.show()
    end_time = time.time()
    print(end_time - start_time)


# check the number of each set, balance
def class_num():
    from event_stream import find_path
    import os
    root_dir, train_path, test_path = find_path()
    dict_loc = [train_path, test_path]
    dict_name = ['train', 'test']
    # for d in dict_loc:
    for d in range(2):
        if d == 0:
            y = np.load(os.path.join(dict_loc[d], "Aug_dataset_labels_remove.npy"), allow_pickle=True)
            x = np.load(os.path.join(dict_loc[d], "Aug_dataset_features_remove.npy"), allow_pickle=True)
            # y = np.load(os.path.join(dict_loc[d], "Ori_dataset_labels_remove.npy"), allow_pickle=True)
            # x = np.load(os.path.join(dict_loc[d], "Ori_dataset_features_remove.npy"), allow_pickle=True)
        else:
            y = np.load(os.path.join(dict_loc[d], "dataset_labels_remove.npy"), allow_pickle=True)
            x = np.load(os.path.join(dict_loc[d], "dataset_features_remove.npy"), allow_pickle=True)
        print(y.shape)
        print(y)
        for i in range(10):
            h = 0
            for j in range(len(y)):
                if y[j] == i:
                    h += 1
            print('{0}: class {1} have {2} data'.format(dict_name[d], i, h))
        print('{0}: feature shape {1}'.format(dict_name[d], x.shape))


# def check_mean_value():
#     from event_stream import find_path
#     import os
#     root_dir, train_path, test_path = find_path()
#     for j in range(2):
#         if j ==0:
#
#         for i in range()
#         y = np.load(os.path.join(j, "{}.npy".format(i)), allow_pickle=True)


# #########$$$$$$$$$$$$$$$$visualize event lip frame from dataset

def lip_img_show(train: bool, n):
    import DVS_Lip
    path = DVS_Lip.path_list(train)
    present_path = path.present_path()
    import matplotlib.pyplot as plt
    if train:
        dataset_name = "Train"
    else:
        dataset_name = "Class"
    tags = ['Positive', 'Negative']
    # cmap = ['OrRd', 'BuPu']
    # cmap = 'magma'
    cmap = 'viridis'

    ###################show a given frame, before stack
    output_arr = np.load(present_path + '{0}.npy'.format(n), allow_pickle=True)
    print(output_arr.shape)

    label_list = np.load(present_path + "dataset_labels.npy", allow_pickle=True)
    label = label_list[n]
    print("Class {}".format(label))
    # ######################display negative and positive###############
    fig, ax = plt.subplots(1, 2)
    plt.suptitle("{0} set {1}, Class {2}".format(dataset_name, n, label))
    ax[0].imshow(output_arr[:, :, 0], cmap=cmap)
    ax[0].set_title(tags[0])
    ax[1].imshow(output_arr[:, :, 1], cmap=cmap)
    ax[1].set_title(tags[1])
    fig.tight_layout()
    # fig.colorbar
    # print(output_arr[:, :, 0]*1E5)
    # fig.show()
    # ##########dispaly neg/positive end##############
    # import matplotlib.pyplot as plt
    plt.imshow(output_arr[:, :, 1], cmap=cmap)
    plt.colorbar(plt.imshow(output_arr[:, :, 1]), cmap=cmap)
    plt.show()


# #########$$$$$$$$$$$$$$$$visualize event lip frame from dataset
from DVS_Animals import AnimalsDvsSliced
from torch.utils.data import Dataset


class animal_img:
    def __init__(self):
        self.dataPath = "C:/Users/ASUS/OneDrive - Nanyang Technological University/datasets/DVSAnimals/data/"

    def animals_img_show(self, n, active_stack_data: bool, display_polar: bool):
        import matplotlib.pyplot as plt
        tags = ['Positive', 'Negative']
        # cmap = ['OrRd', 'BuPu']
        # cmap = 'magma'
        cmap = 'viridis'
        axis_labels = ['cat', 'dog', 'camel', 'cow', 'sheep', 'goat', 'wolf',
                       'squirrel', 'mouse', 'dolphin', 'shark', 'lion', 'monkey',
                       'snake', 'spider', 'butterfly', 'bird', 'duck', 'zebra']
        if active_stack_data:
            # ###################show a given frame, after stack
            output_arr_0 = np.load(self.dataPath + "events/x_train.npy", allow_pickle=True)
            output_arr = output_arr_0[n, :, :, :]
            label_list = np.load(self.dataPath + "events/y_train.npy", allow_pickle=True)
            label = label_list[n]
        else:
            ###################show a given frame, before stack
            output_arr = np.load(self.dataPath + 'events/{0}.npy'.format(n), allow_pickle=True)
            label_list = np.load(self.dataPath + "events/dataset_labels.npy", allow_pickle=True)
            label = label_list[n]
        # ######################display negative and positive###############
        fig, ax = plt.subplots(1, 2)
        plt.suptitle("number {0}, Class {1}, {2}".format(n, int(label),
                                                         axis_labels[int(label)]))
        ax[0].imshow(output_arr[:, :, 0], cmap=cmap)
        ax[0].set_title(tags[0])
        ax[1].imshow(output_arr[:, :, 1], cmap=cmap)
        ax[1].set_title(tags[1])
        fig.tight_layout()

        # print(output_arr[:, :, 0]*1E5)
        fig.show()
        # ##########dispaly neg/positive end##############
        # import matplotlib.pyplot as plt
        plt.imshow(output_arr[:, :, 1], cmap=cmap)
        plt.colorbar(plt.imshow(output_arr[:, :, 1]), cmap=cmap)
        if display_polar:
            plt.show()

    def plot_scatter(self, n, aspect, part_e_s: bool):

        dataset = AnimalsDvsSliced(self.dataPath)
        events, class_name, class_index = dataset.__getitem__(index=n)

        t_pos, x_pos, y_pos, t_neg, x_neg, y_neg = self.get_scatter(events, part_e_s)
        # fig = plt.figure(figsize=(20, 15))
        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(projection='3d')

        try:
            ax.set_box_aspect(aspect)
        except:
            ax.set_box_aspect((2, 1, 1))
        cmp = ['autumn_r', 'winter_r']

        ax.scatter(t_pos, x_pos, y_pos, label='Positive', cmap=cmp[0], c=t_pos, s=2, alpha=0.2)
        ax.scatter(t_neg, x_neg, y_neg, label='Negative', cmap=cmp[1], c=t_neg, s=2, alpha=0.2)

        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        plt.title("simple 3D scatter plot, number {0}, class {1}".format(n, class_name))
        ax.set_xlabel('Time', fontdict=dict(weight='bold', fontsize=80))
        plt.show()

    def get_scatter(self, events, part_e_s: bool):
        x = tuple(events["x"])
        y = tuple(events["y"])
        t = tuple(events["t"])
        p = tuple(events["p"])
        x_pos = []
        y_pos = []
        t_pos = []
        x_neg = []
        y_neg = []
        t_neg = []
        if part_e_s:
            #     zoom in 2s
            entire_t = (t[-1] - t[0]) * 1E-6
            if entire_t <= 1:
                print("error: shorter than 2s")
            else:
                begin_n = int((len(t) - len(t) * 2 / int(entire_t)) // 2)
                end_n = int(len(t) - begin_n)
                t = t[begin_n:end_n]
                x = x[begin_n:end_n]
                y = y[begin_n:end_n]
                p = p[begin_n:end_n]
        for i in range(len(t)):
            t_n = (t[i] - t[0]) * 1E-6
            if t_n < 2:
                print(t_n)
                if p[i]:
                    x_pos.append(x[i])
                    y_pos.append(y[i])
                    t_pos.append(t_n)
                else:
                    x_neg.append(x[i])
                    y_neg.append(y[i])
                    t_neg.append(t_n)
            else:
                break
        return t_pos, x_pos, y_pos, t_neg, x_neg, y_neg
