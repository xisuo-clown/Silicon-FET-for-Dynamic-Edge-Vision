###############IMPORTENT!!!!!!################
# 1. split into 2 seconds part: according to the fast decay memory and the movement speed
# 2. nearly no difference from applying split into 5 parts(with overlapping) / training data
# to split into 3 parts(without overlapping)
# the only thing influence the accuracy is the data augmentation.
# 3. the length of event stream for training and testing can be shorter.
# (but should be larger than length of 1 cycle)
# some of the gestures are under different frequency.
# the accuracy is strongly rely on the memory length/gestures' frequency.
# ############################################

import numba as nb
import math
import random

import numpy as np
import IPython
from keras import Input

from keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Layer, Add, \
    Dropout
from keras.models import Model
import tensorflow as tf
from keras.regularizers import l2
import matplotlib.pyplot as plt
import itertools

from event_stream import n_num


######CONFIGURATION################
# Path #####################
def find_path(suffix: str):
    # Original event stream file path
    root_dir = '/usr1/home/s124mdg41_03/Integrated_package/DvsGes'
    # path for event array saving: train and test
    train_save_path = (root_dir + '/event_array/train_set_' + suffix + '/')
    test_save_path = (root_dir + '/event_array/test_set_' + suffix + '/')
    import os
    os.makedirs(train_save_path, exist_ok=True)
    os.makedirs(test_save_path, exist_ok=True)
    # train_save_path = (root_dir + '/CBRAM/train_set_eve/')
    # test_save_path = (root_dir + "/CBRAM/test_set_eve/")
    return root_dir, train_save_path, test_save_path


# Parameters ################
# coefficient: rescal time axis  e.g. 1s => c*1s
# c = 2E-4
c = 3E-4

# c = 1E-5
# the pulse interval (in pn vs id curve)
t_interval = 2E-4

# 1. n_clip: decide the length of each part (split the 6-second event stream into n_clip shorter event stream)
# (e.g. 2s event stream => n_clip = 3)
n_clip = 3
# 2. n_num: sum of divided parts(5=>split into 5 parts)!!!!!!!when change the n_num, the n_step may need to reset
# n_num = 5
n_num = 3
# 3. n_step: the step (the gap of the events number from the beginning of present
# part to the beginning of next part) (e.g. 6=>total events/6 as step length)
n_step = 6


# Fitting Formula
def para_transistor_exp():
    # time(-1.4v, 10ms)
    # a1, a2, tau1, tau2 = 2.54079, 0.58651, 2.58652E-5, 5.46109E-4
    a1, a2, tau1, tau2 = 1.98088E-4, 0.04903, 8.41192E-7, 1.98484
    # pulse number(-1.4v, 200us)
    # unit: uA
    t1, y1 = -57.98876, 9.46142E-5
    return a1, a2, tau1, tau2, t1, y1


def para_transistor_bi_exp():
    # relaxation
    a1, a2, tau1, tau2 = 2.54079, 0.58651, 2.58652E-5, 5.46109E-4
    # pulse number(-1.4v, 200us)
    b1, b2, t1, t2, y1 = -0.34566, -0.31446, 1.29858, 29.16395, 2.18168

    return a1, a2, tau1, tau2, b1, b2, t1, t2, y1


# #######################################################################


# Initialization: Load event stream
def init_event(spikingjelly=None, suffix="testing"):
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture as D128G
    root_dir, train_save_path, test_save_path = find_path(suffix)
    train_set_eve = D128G(root_dir, train=True, data_type='event')
    test_set_eve = D128G(root_dir, train=False, data_type='event')
    return train_set_eve, test_set_eve


# _set_eve contains x,y,t,p


# Generate a dictionary,
# For each event video, save events happened in the pixels.
# total 16384*2 (128*128*2) lists
def polar_save_as_list(set_eve, n, indexarr):
    event, label = set_eve[n]
    x0 = tuple(event["x"])
    y0 = tuple(event["y"])
    t = tuple(event["t"])
    p = tuple(event["p"])
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
    for j in range(n_num):
        n = j * int(len(t) / n_step)
        ne = n + int(len(t) / n_clip) - 1
        # print('n is {}'.format(n))
        # print('ne is {}'.format(ne))
        end_t = (t[ne] - t[n]) * 1E-6 * c
        for h in range(128 * 128):
            events_dict_pos[h] = []
            events_dict_neg[h] = []
            events_dict_pos_time[h] = []
            events_dict_neg_time[h] = []
            # print(len(t))
        for i in range(len(t)):
            key = indexarr[x0[i]][y0[i]]
            if ne > i > n:
                if p[i] != 1:
                    events_dict_neg[key].append((t[i] - t[n]) * 1E-6 * c)
                else:
                    events_dict_pos[key].append((t[i] - t[n]) * 1E-6 * c)
                if end_t > (t[ne] - t[n]):
                    print('error{}'.format(key))
        end_time.append(end_t)

        dict_pos[j] = events_dict_pos
        dict_neg[j] = events_dict_neg
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
        dict_pos_time[j] = events_dict_pos_time
        dict_neg_time[j] = events_dict_neg_time

    # return dict_pos, dict_neg, label, end_time
    return dict_pos_time, dict_neg_time, label


# Save the positive & negative, generate 30 samples from per video
def polarity_process_transistor_match(train: bool):
    global n_num
    n_num = 3
    import time
    begin = time.time()
    indexarr = index_arr()
    train_set_eve, test_set_eve = init_event()
    root_dir, train_save_path, test_save_path = find_path()
    id_list, id_pulse_dict = generate_comparison_idpd_index()
    if train:
        set_eve = train_set_eve
        data_num = 1176
        save_path = train_save_path
    else:
        set_eve = test_set_eve
        data_num = 288
        save_path = test_save_path
    label_arr = []
    a = 0
    from tqdm import tqdm
    import numpy as np
    pos_temp_save = []
    neg_temp_save = []
    temp_save = [pos_temp_save, neg_temp_save]

    a1, a2, tau1, tau2, b1, b2, t1, t2, y1 = para_transistor_bi_exp()

    for n in tqdm(range(data_num), desc="process"):
        output_arr = np.empty((128, 128, 2))
        event, label = set_eve[n]
        if label != 2:
            dict_pos_time, dict_neg_time, label = polar_save_as_list(set_eve, n, indexarr)
            dict_list = [dict_pos_time, dict_neg_time]
            # each class, generate 30 frames
            # for d in range(30):
            for d in range(n_num):
                label_arr.append(label)
                # pos_events_dict = dict_pos[d]
                # neg_events_dict = dict_neg[d]
                for polar in range(2):
                    temp_save[polar] = []
                    events_dict = dict_list[polar][d]
                    # print("event list is {}".format(events_dict))
                    for i in range(128 * 128):
                        if events_dict[i]:
                            i_last = id_pulse_dict[1]
                            for j in events_dict[i]:
                                if j != events_dict[i][-1]:
                                    i_last = take_closest(id_list,
                                                          id_time(i_last, j, a1, a2, tau1, tau2), id_pulse_dict)
                                else:
                                    i_last = id_time(i_last, j, a1, a2, tau1, tau2)
                            temp_save[polar].append(i_last)
                        else:
                            temp_save[polar].append(0)
                    for k in range(128):
                        for m in range(128):
                            output_arr[k, m, polar] = temp_save[polar][int(indexarr[m][k])]
                # print(output_arr.shape)
                np.save(save_path + "{0}.npy".format(a), output_arr)
                a += 1
        # here we removed class 2 (other gestures)
        np.save(save_path + "dataset_labels.npy", label_arr)
    print("{0} length of data {1}".format(data_num, len(label_arr)))
    print(label_arr)
    end = time.time()
    print(end - begin)
    # Events stream transform into array in n.npy, print arrays to visualize frames
    return


# !!!!!!!!!!
# Save the positive & negative, generate 30 samples from per video
def polarity_process_transistor_conditions(train: bool, para_before_tune, para_after_tune, suffix: str,
                                           tune_choice: list):
    import time
    begin = time.time()
    indexarr = index_arr()
    train_set_eve, test_set_eve = init_event(suffix)
    root_dir, train_save_path, test_save_path = find_path(suffix)
    if train:
        set_eve = train_set_eve
        data_num = 1176
        save_path = train_save_path
    else:
        set_eve = test_set_eve
        data_num = 288
        save_path = test_save_path
    label_arr = []
    a = 0
    from tqdm import tqdm
    import numpy as np
    pos_temp_save = []
    neg_temp_save = []
    temp_save = [pos_temp_save, neg_temp_save]

    # y0, A1, A2, A3, t1, t2, t3, d_0, l_a, l_b, id_0 = transistor_3_exp()
    i_d_before_tune = para_before_tune
    i_d_after_tune = para_after_tune
    i_d_table = [i_d_before_tune, i_d_after_tune]
    for n in tqdm(range(data_num), desc="process"):
        output_arr = np.empty((128, 128, 2))
        event, label = set_eve[n]
        if label != 2:
            i_d = i_d_table[int(tune_choice[label])]

            dict_pos_time, dict_neg_time, label = polar_save_as_list(set_eve, n, indexarr)
            dict_list = [dict_pos_time, dict_neg_time]
            # each class, generate 30 frames
            # for d in range(30):
            for d in range(n_num):
                label_arr.append(label)
                # pos_events_dict = dict_pos[d]
                # neg_events_dict = dict_neg[d]
                for polar in range(2):
                    temp_save[polar] = []
                    events_dict = dict_list[polar][d]
                    # print("event list is {}".format(events_dict))
                    for i in range(128 * 128):
                        if events_dict[i]:

                            id_last = i_d.d[0]
                            for j in events_dict[i]:

                                y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_, l_a, l_b = i_d.get_para(id_last)

                                id_b, id_a = id_time_new(id_last, j, y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_, l_a, l_b)

                                if j != events_dict[i][-1]:
                                    id_last = id_a

                                else:
                                    id_last = id_b

                            if id_last - 15.2 > 0:
                                temp_save[polar].append(id_last - 15.2)

                            else:
                                temp_save[polar].append(0)
                            # # >>>>>>>>>>??????????????????????
                            # if id_last - 15.2 < 0:
                            #     print(id_last)

                            # temp_save[polar].append(id_last)
                        else:
                            temp_save[polar].append(0)
                    for k in range(128):
                        for m in range(128):
                            output_arr[k, m, polar] = temp_save[polar][int(indexarr[m][k])]
                # print(output_arr.shape)
                np.save(save_path + "{0}.npy".format(a), output_arr)
                a += 1
        # here we removed class 2 (other gestures)
        np.save(save_path + "dataset_labels.npy", label_arr)
    print("{0} length of data {1}".format(data_num, len(label_arr)))
    print(label_arr)
    end = time.time()
    print(end - begin)
    # Events stream transform into array in n.npy, print arrays to visualize frames
    return


from tqdm import tqdm
import numpy as np
import time


# def polarity_process_transistor_cal(train: bool):
#     begin = time.time()
#     indexarr = index_arr()
#     train_set_eve, test_set_eve = init_event()
#     root_dir, train_save_path, test_save_path = find_path()
#     a1, a2, tau1, tau2, t1, y1 = para_transistor_exp()
#     if train:
#         set_eve = train_set_eve
#         data_num = 1176
#         save_path = train_save_path
#     else:
#         set_eve = test_set_eve
#         data_num = 288
#         save_path = test_save_path
#     label_arr = []
#     a = 0
#     for n in tqdm(range(data_num), desc="process"):
#         output_arr = np.empty((128, 128, 2))
#         temp_save = []
#         dict_pos_time, dict_neg_time, label = polar_save_as_list(set_eve, n, indexarr)
#         dict_list = [dict_pos_time, dict_neg_time]
#         # here we removed class 2 (other gestures)
#         if label != 2:
#             # each class, generate 30 frames
#             # for d in range(30):
#             for d in range(n_num):
#                 label_arr.append(label)
#                 for q in range(2):
#                     events_dict_time = dict_list[q][d]
#                     # print(events_dict)
#                     # print(end_time[d])
#                     for i in range(128 * 128):
#                         if events_dict_time[i]:
#                             i_last = 0.00000259
#                             for j in range(len(events_dict_time[i]) - 1):
#                                 t = events_dict_time[i][j]
#                                 id_b_p = id_time(i_last, t, a1, a2, tau1, tau2)
#                                 i_last = id_num_exp(id_b_p, t1, y1)
#                             t = events_dict_time[i][len(events_dict_time[i]) - 1]
#                             id_b_p = id_time(i_last, t, a1, a2, tau1, tau2)
#                             # print(id_b_p)
#                             temp_save.append(id_b_p)
#                         else:
#                             temp_save.append(0)
#                     for k in range(128):
#                         for m in range(128):
#                             output_arr[k, m, q] = temp_save[int(indexarr[m][k])]
#                 # print(output_arr.shape)
#                 np.save(save_path + "{0}.npy".format(a), output_arr)
#                 a += 1
#     np.save(save_path + "dataset_labels.npy", label_arr)
#     end = time.time()
#     print(end - begin)
#     # Events stream transform into array in n.npy, print arrays to visualize frames
#     return


# Supplementary###########################################################
# Generate an (128,128) array for mapping
def index_arr():
    import numpy as np
    h = 0
    indexarr = np.zeros((128, 128))
    for i in range(128):
        for j in range(128):
            indexarr[i][j] = h
            h += 1
    return indexarr


# Output a dictionary, saving pulse number and corresponding
def generate_comparison_idpd_index():
    # interval/pulse width = 100us, V = 1.2v
    a1, a2, tau1, tau2, b1, b2, t1, t2, y1 = para_transistor_bi_exp()
    # save present id
    id_pulse_dict = {}
    # save previous id
    id_list = []
    for x in range(1, 100):
        # time is on the us scale
        if x != 1:
            # id num interval = 200us
            # key i: save id (pulse x)
            # list[i]:save id(after i+1 pulses, before i+2 pulses),
            # e.g. list[0] equals to id(after pul1, before pul2)

            ######in this function it produces the value of transistor by time without pulse
            id_list.append(id_time(id_num((x - 1), b1, b2, t1, t2, y1), t_interval, a1, a2, tau1, tau2))
            ######in this function it produces the current value of transistor by successive pulse
            id_pulse_dict[x] = id_num(x, b1, b2, t1, t2, y1)
            # print(id_pulse_dict[x])
        else:
            # key 1: id(when pulse = 1)
            id_list.append(id_time(id_num(1, b1, b2, t1, t2, y1), t_interval, a1, a2, tau1, tau2))
            # key [1], after pulse 1
            id_pulse_dict[1] = id_num(x, b1, b2, t1, t2, y1)
            # print(id_pulse_dict[1])
    return id_list, id_pulse_dict


def take_closest(mylist, mynumber, id_pulse_dict):
    from bisect import bisect_left
    if mynumber > 0.20:
        pos = bisect_left(mylist, mynumber)
        if pos == 0:
            # print('equal to 1 pulse')
            return id_pulse_dict[1]
        elif pos == len(mylist):
            # print('equal to ''{}'.format(len(mylist)) + 'pulses')
            return id_pulse_dict[len(mylist)]
        elif mylist[pos] - mynumber < mynumber - mylist[pos - 1]:
            # print('equal to ''{}'.format(pos + 1) + 'pulses')
            return id_pulse_dict[pos + 1]
        else:
            # print('equal to ''{}'.format(pos) + 'pulses')
            return id_pulse_dict[pos]
    else:
        # print('equal to 1 pulse')
        return id_pulse_dict[1]


@nb.jit(nopython=True)
def id_num(x, b1, b2, t1, t2, y1):
    id_vs_num = b1 * math.exp(-x / t1) + b2 * math.exp(-x / t2) + y1
    return id_vs_num


@nb.jit(nopython=True)
def id_num_exp(i_d, t1, y1):
    # a1, a2, tau1, tau2, t1, y1 = para_transistor_exp()
    i_d = (i_d * (t1 + 1) - y1) / t1
    # print(i_d)
    return i_d


class calculate_match:
    def __init__(self, y0: list, a1: list, t1: list, a2: list, t2: list, a3: list, t3: list,
                 d: list, a: float, b: float, id_0:float=0,id_th:float=0):

        self.y0 = y0
        self.A1 = a1
        self.t1 = t1
        self.A2 = a2
        self.t2 = t2
        self.A3 = a3
        self.t3 = t3

        # self.y0 = [19.91713, 20.34551, 20.67151]
        # self.A1 = [7.52151, 7.61645, 8.42568]
        # self.A2 = [2.72444, 3.31655, 2.87707]
        # self.A3 = [0.34774, 0.42499, 0.54843]
        # self.t1 = [2.56303E-5, 2.3492E-5, 2.46777E-5]
        # self.t2 = [1.70132E-4, 1.50927E-4, 1.81061E-4]
        # self.t3 = [0.39407, 0.10214, 0.47215]

        # d: threshold for shifting the relaxation formula
        self.d = d

        self.a = a

        self.b = b
        self.id_0 = id_0
        self.id_th = id_th
        # self.id_0 = 28.5

    def get_para(self, i_last):
        if i_last >= self.d[2]:
            i = 2
        elif i_last >= self.d[1]:
            i = 1
        else:
            i = 0
        l_a, l_b = self.a, self.b
        y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_ = (self.y0[i], self.A1[i], self.A2[i],
                                                 self.A3[i], self.t1[i], self.t2[i],
                                                 self.t3[i], self.d[i])
        return y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_, l_a, l_b


@nb.jit(nopython=True)
def id_time_new(i_last, t, y0, A1, A2, A3, t1, t2, t3, d, l_a, l_b):
    id_b = (i_last / d) * (A1 * math.exp(-t / t1) + A2 * math.exp(-t / t2) + A3 * math.exp(-t / t3) + y0)
    # print(id_vs_time)
    id_a = l_a * id_b + l_b
    if id_a > 30.5:
        id_a = 30.5
    return id_b, id_a


# @nb.jit(nopython=True)
# def id_time_new(i_last, t, y0, A1, A2, A3, t1, t2, t3, d, l_a, l_b):
#     # for positive pulse, id_vs_time is the absolute value
#     # a1, a2, tau1, tau2, t1, y1 = para_transistor_exp()
#     # switch between 3 bi-decay formulas
#     # id varies: from28-30(uA)
#     # y0 = [19.91713, 20.34551, 20.67151]
#     # A1 = [7.52151, 7.61645, 8.42568]
#     # t1 = [2.56303E-5, 2.3492E-5, 2.46777E-5]
#     # A2 = [2.72444, 3.31655, 2.87707]
#     # t2 = [1.70132E-4, 1.50927E-4, 1.81061E-4]
#     # A3 = [0.34774, 0.42499, 0.54843]
#     # t3 = [0.39407, 0.10214, 0.47215]
#     # d = [28.67, 29.54, 30.44]
#     if i_last > 30:
#         i = 2
#     elif i_last > 29:
#         i = 1
#     else:
#         i = 0
#     id_vs_time = (i_last / d[i]) * (A1[i] * math.exp(-t / t1[i]) +
#                                     A2[i] * math.exp(-t / t2[i]) +
#                                     A3[i] * math.exp(-t / t3[i]) + y0[i])
#     # print(id_vs_time)
#     next_id = l_a * id_vs_time + l_b
#     return id_vs_time, next_id


@nb.jit(nopython=True)
def id_time(i_last, t, a1, a2, tau1, tau2):
    # for positive pulse, id_vs_time is the absolute value
    # a1, a2, tau1, tau2, t1, y1 = para_transistor_exp()
    id_vs_time = i_last * (a1 * math.exp(-t / tau1) + a2 * math.exp(-t / tau2)) / (a1 + a2)
    # print(id_vs_time)
    return id_vs_time


def calculate_id(i_d, t, last: bool):
    if last:
        i_d = id_time(i_d, t)
    else:
        id_b_p = id_time(i_d, t)
        i_d = id_num_exp(id_b_p)
    return i_d


def gen_augmentation_frame(suffix: str):
    import os
    import numpy as np
    import time
    begin = time.time()
    root_dir, train_path, test_path = find_path(suffix)
    from tqdm import tqdm
    from sklearn.preprocessing import LabelEncoder
    # Augmentation: the training set (aug.npy and label index)
    y_labelencoder = LabelEncoder()
    y = np.load(os.path.join(train_path, "dataset_labels.npy"), allow_pickle=True)

    y_train = y_labelencoder.fit_transform(y)
    y_train = y_train.tolist()

    for i in tqdm(range(len(y_train)), desc="Rotate and shift"):
        path = os.path.join(train_path, f'{i}.npy')
        x_train = np.load(path, allow_pickle=True)
        x, y = aug_process(x_train, y_train[i])
        # os.remove(path)
        if i == 0:
            y_list = y
        else:
            y_list = np.concatenate((y_list[:, ], y[:, ]), axis=0)
        np.save(train_path + "Aug_{}.npy".format(i), x)
    print('y_train shape: should be {0}, len_y is {1}'.format(6 * len(y_train), y_list.shape))
    np.save(train_path + "Aug_dataset_labels.npy", y_list)
    # stack the frames: train and test
    gen_stack_frame(Aug=True, suffix=suffix)
    print('END: Stacking')
    end = time.time()
    print(end - begin)


def aug_process(x_train, y_train: int):
    from keras.layers import RandomRotation, RandomTranslation
    from keras import Sequential
    from numpy import expand_dims, row_stack, empty
    x_rota1 = RandomRotation(factor=(-0.1, 0), fill_mode='reflect')
    x_rota2 = RandomRotation(factor=(0, 0.1), fill_mode='reflect')
    x_shift = RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1),
                                fill_mode='reflect', fill_value=0.0)
    x_shift_1 = Sequential([
        RandomRotation(factor=(-0.1, 0), fill_mode='constant'),
        RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1),
                          fill_mode='constant', fill_value=0.0)])
    x_shift_2 = Sequential([
        RandomRotation(factor=(0, 0.1), fill_mode='constant'),
        RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1),
                          fill_mode='constant', fill_value=0.0)])
    y = empty(6)
    y.fill(int(y_train))
    x = expand_dims(x_train, axis=0)
    x = row_stack((x, expand_dims(x_rota1(x_train), axis=0)))
    x = row_stack((x, expand_dims(x_rota2(x_train), axis=0)))
    x = row_stack((x, expand_dims(x_shift(x_train), axis=0)))
    x = row_stack((x, expand_dims(x_shift_1(x_train), axis=0)))
    x = row_stack((x, expand_dims(x_shift_2(x_train), axis=0)))
    # y_list = row_stack((y_list, y))
    return x, y


def gen_stack_frame(Aug: bool, suffix):
    import os
    import numpy as np
    import time
    begin = time.time()
    root_dir, train_path, test_path = find_path(suffix)
    path_type = [train_path, test_path]
    # processing
    import tensorflow as tf
    from numpy import expand_dims
    from tqdm import tqdm
    name_type = ['Train', 'Test']
    if Aug:
        name_load = ['Aug_', '']
        name_save = ['Aug_', '']
        a = 6

    else:
        name_load = ['', '']
        name_save = ['Ori_', '']
        a = 1
    # here we can choose, to stack the original frames, or the augmented frames

    from sklearn.preprocessing import LabelEncoder
    y_labelencoder = LabelEncoder()
    for j in range(2):
        y = np.load(os.path.join(path_type[j], "dataset_labels.npy"), allow_pickle=True)
        print("y shape is {}".format(y.shape))
        if j == 1:
            a = 1
        y = y_labelencoder.fit_transform(y)
        y = y.tolist()
        for i in tqdm(range(len(y)), desc="Stack_{}".format(name_type[j])):
            path = os.path.join(path_type[j], f'{name_load[j]}{i}.npy')
            x = np.load(path, allow_pickle=True)
            if x.ndim < 4:
                x = expand_dims(x, axis=0)
            y_1 = np.full(a, y[i])

            # stack (creating a initial array)
            if i != 0:
                x_0[i * a:(i + 1) * a, :, :, :] = x
                y_0[i * a:(i + 1) * a, ] = y_1
                # print(y_0.shape)
            else:
                x_0 = np.zeros((len(y) * a, 128, 128, 2))
                y_0 = np.zeros((len(y) * a,))
                x_0[0:a, :, :, :] = x
                y_0[0:a, ] = y_1
            # os.remove(path)
        print('fianl x shape: {0}, and len_y is {1}'.format(x_0.shape, len(y_0)))

        # stack (without creating a initial array)
        #     if i != 0:
        #         x_0 = np.row_stack((x_0, x))
        #         y_0 = np.concatenate((y_0[:,],y_1[:,]),axis=0)
        #         # print(y_0.shape)
        #     else:
        #         x_0 = x
        #         y_0 = y_1
        #     # os.remove(path)
        # print('fianl x_train shape: {0}, and len_y is {1}'.format(x_0.shape, len(y_0)))

        np.save(path_type[j] + "{}dataset_features.npy".format(name_save[j]), x_0)
        np.save(path_type[j] + "{}dataset_labels.npy".format(name_save[j]), y_0)
        print("{0}, labels: {1}".format(name_type, y))
    print('END: Stacking')
    end = time.time()
    print(end - begin)


def save_params_to_file(file_path, **kwargs):
    with open(file_path, 'w') as file:
        for key, value in kwargs.items():
            file.write(f"{key} = {value}\n")


def polar_remove_set7(aug, suffix: str):
    import os
    root_dir, train_path, test_path = find_path(suffix)
    # processing in
    import numpy as np
    x_test = np.load(os.path.join(test_path, "dataset_features.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(test_path, "dataset_labels.npy"), allow_pickle=True)
    if aug:
        name = 'Aug_'
    else:
        name = 'Ori_'
    x_name = "{}dataset_features.npy".format(name)
    y_name = "{}dataset_labels.npy".format(name)
    x_train = np.load(os.path.join(train_path, x_name), allow_pickle=True)
    y_train = np.load(os.path.join(train_path, y_name), allow_pickle=True)
    std = np.max(x_train)
    x_train = x_train / std
    x_test = x_test / std
    a, b = 0, 0
    print('the original test shape {0}, label{1}'.format(x_test.shape, y_test.shape))
    print('the original train shape {0}, label {1}'.format(x_train.shape, y_train.shape))
    for i in range(len(y_train)):
        if y_train[i] == 7:
            if y_train[i - 1] != 7:
                a = i
                print('train begin', i)
            elif y_train[i + 1] != 7:
                b = i + 1
                print('train end', i + 1)
                break
    print((b - a) / 2)
    c, d = 0, 0
    for i in range(len(y_test)):
        if y_test[i] == 7:
            if y_test[i - 1] != 7:
                c = i
                print('test begin', i)
            elif y_test[i + 1] != 7:
                d = i + 1
                print('test end', i + 1)
                break
    print((d - c) / 2)
    x_train = np.concatenate([x_train[:int(a + (b - a) / 2), :, :, :], x_train[b:, :, :, :]], axis=0)
    y_train = np.concatenate((y_train[:int(a + (b - a) / 2)], y_train[b:]), axis=0)

    x_test = np.concatenate((x_test[:int(c + (d - c) / 2), :, :, :], x_test[d:, :, :, :]), axis=0)
    y_test = np.concatenate((y_test[:int(c + (d - c) / 2)], y_test[d:]), axis=0)
    print('y train', y_train.shape)
    print('x train', x_train.shape)
    print('y test', y_test.shape)
    print('x test', x_test.shape)
    np.save(os.path.join(test_path, "dataset_features_remove.npy".format(name)), x_test)
    np.save(os.path.join(test_path, "dataset_labels_remove.npy".format(name)), y_test)
    np.save(os.path.join(train_path, "{0}dataset_features_remove.npy".format(name)), x_train)
    np.save(os.path.join(train_path, "{0}dataset_labels_remove.npy".format(name)), y_train)
    print('Pack and remove_end')


class ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int, k: float, down_sample=False):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__down_sample = down_sample
        self.__strides = [2, 1] if down_sample else [1, 1]
        self.__k = k

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()
        self.conv_2 = Conv2D(self.__channels, strides=self.__strides[1],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_2 = BatchNormalization()
        self.merge = Add()

        if self.__down_sample:
            # perform down sampling using stride of 2, according to [1].
            self.res_conv = Conv2D(
                self.__channels, strides=2, kernel_size=(1, 1), kernel_initializer=INIT_SCHEME, padding="same")
            self.res_bn = BatchNormalization()

    def call(self, inputs):
        res = inputs

        x = self.conv_1(inputs)
        x = self.bn_1(x)
        x = tf.nn.relu(x)
        x = self.conv_2(x)
        x = self.bn_2(x)

        if self.__down_sample:
            res = self.res_conv(res)
            res = self.res_bn(res)
            res = self.__k * res

        # if not perform down sample, then add a shortcut directly
        x = self.merge([x, res])
        out = tf.nn.relu(x)
        return out


class ResNet18(Model):

    def __init__(self, num_classes, **kwargs):
        """
            num_classes: number of classes in specific classification task.
        """
        super().__init__(**kwargs)

        # control current simultaneously
        filters = [32, 64, 64]
        k = [0.8, 0.9, 1, 1, 1, 1]
        # k = [1, 1, 1, 1, 1, 1]
        # previous function
        self.conv_1 = Conv2D(filters[0],
                             (4, 4),
                             strides=2,
                             padding="same", kernel_initializer="he_normal")
        # now function
        # self.conv_1 = Conv2D(filters[0],
        #                      (4, 4),
        #                      strides=2,
        #                      padding="same", kernel_initializer="ones")
        self.init_bn = BatchNormalization()
        self.pool_2 = MaxPool2D(
            pool_size=(2, 2),
            strides=2,
            padding="same")

        self.res_1_1 = ResnetBlock(filters[0], k[0])
        self.res_1_2 = ResnetBlock(filters[0], k[1])

        self.res_2_1 = ResnetBlock(filters[1], k[1], down_sample=True)
        self.res_2_2 = ResnetBlock(filters[1], k[2])

        self.res_3_1 = ResnetBlock(filters[2], k[4], down_sample=True)
        self.res_3_2 = ResnetBlock(filters[2], k[5])

        self.res_4_1 = ResnetBlock(filters[2], k[4], down_sample=True)
        self.res_4_2 = ResnetBlock(filters[2], k[5])

        self.avg_pool = GlobalAveragePooling2D()
        self.merge = Add()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")
        # self.fc = Dense(10, activation="softmax")

    def call(self, inputs):
        out = self.conv_1(inputs)
        out = self.init_bn(out)
        from tensorflow.keras.layers import Input, ReLU, Dense
        out = ReLU()(out)
        # out = tf.nn.relu(out)
        # # # ######without extra shortcut
        # out = self.pool_2(out)
        # for res_block in [self.res_1_1,
        #                   self.res_1_2,
        #                   self.res_2_1,
        #                   self.res_2_2,
        #                   self.res_3_1,
        #                   self.res_3_2,
        #                   self.res_4_1,
        #                   self.res_4_2
        #                   ]:
        #     out = res_block(out)
        # # # #####without extra shortcut end

        # #####with extra shortcut
        out_0 = self.pool_2(out)
        out = self.res_1_1(out_0)
        out = self.res_1_2(out)
        out = self.merge([out, out_0])
        out = self.res_2_1(out)

        # dvs gesture: remove last 3

        # # dvs animals remove res3_1
        #
        # out = self.res_2_2(out)
        # out = self.res_3_1(out)
        # out = self.res_3_2(out)
        ####with extra shortcut end

        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

    def build_graph(self):
        x = Input(shape=(128, 128, 2))
        return Model(inputs=[x], outputs=self.call(x))


def hyper_tuner_for_times(aug, tune, model_path, times: int, dir_name: str, suffix: str):
    random_state = random.randint(1, 1000)
    train_validation_rate = random.uniform(0.1, 0.2)
    initial_learning_rate = random.uniform(0.01, 0.1)
    decay_steps = random.randint(1000, 5000)
    decay_rate = random.uniform(0.1, 1)
    STEPS = random.randint(100, 200)
    from datetime import datetime
    if tune:
        name = datetime.now().strftime("%Y%m%d_%H%M%S") + "_tuned"
    else:
        name = datetime.now().strftime("%Y%m%d_%H%M%S")
    for i in range(times):
        hyper_tuner(aug, tune, model_path, name + f"_{i}", random_state, train_validation_rate, initial_learning_rate,
                    decay_steps, decay_rate, STEPS, dir_name, suffix)


def polar_remove_load(aug, random_state=86, train_validation_rate=0.125, suffix="str"):
    import os
    root_dir, train_path, test_path = find_path(suffix)
    # processing
    import numpy as np
    if aug:
        name = 'Aug_'
    else:
        name = 'Ori_'

    x_test = np.load(os.path.join(test_path, "dataset_features_remove.npy".format(name)), allow_pickle=True)
    y_test = np.load(os.path.join(test_path, "dataset_labels_remove.npy".format(name)), allow_pickle=True)
    x_train = np.load(os.path.join(train_path, "{0}dataset_features_remove.npy".format(name)), allow_pickle=True)
    y_train = np.load(os.path.join(train_path, "{0}dataset_labels_remove.npy".format(name)), allow_pickle=True)
    # print(x_train.shape)
    # print(y_train)
    from sklearn.preprocessing import LabelEncoder
    y_labelencoder = LabelEncoder()
    y_train = y_labelencoder.fit_transform(y_train)
    y_test = y_labelencoder.fit_transform(y_test)
    y_list = [y_train, y_test]
    x_list = [x_train, x_test]
    name_type = ['train', 'test']
    print(x_test.shape, x_train.shape)
    mean_train = []
    mean_test = []
    mean = [mean_train, mean_test]

    for i in range(2):
        y = y_list[i]
        x = x_list[i]
        for h in range(10):
            a = 0
            for j in range(len(y)):
                if h == y[j]:
                    a += 1
                    b = j + 1
            print('mean {0} class {1}, type {2}'.format(np.mean(x[b - a:b, :, :, :]), h, name_type[i]))
            mean[i].append(np.mean(x[b - a:b, :, :, :]))
    for i in range(10):
        print('{0} gap {1}'.format(i, mean_train[i] - mean_test[i]))
        print('{}'.format((mean_train[i] - mean_test[i]) / mean_train[i]))
    # for i in range(len(y_test)):
    #     if

    # x_train, y_train = shuffle(x_tra, y_tra, random_state=seed)
    seed = 42
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = (
        train_test_split(x_train, y_train, test_size=train_validation_rate, random_state=random_state))
    return x_train, x_test, y_train, y_test, x_val, y_val


# #########build model############################################
def hyper_tuner(aug, tune, model_path, name, random_state, train_validation_rate, initial_learning_rate, decay_steps,
                decay_rate, STEPS, dir_name, suffix):
    import time
    start = time.time()
    import random
    txt_data = {}
    txt_name = "20241103_002953"
    # with open(f"./{dir_name}/{txt_name}.txt", "r", encoding="utf-8") as file:
    #     for line in file:
    #         if "accuracy" not in line and "tune" not in line:
    #             txt_data[line.split(":")[0]]=line.split(":")[1][:-2].strip()
    #             if "." in line.split(":")[1][:-2].strip():
    #                 txt_data[line.split(":")[0]]=float(txt_data[line.split(":")[0]])
    #             else:
    #                 txt_data[line.split(":")[0]] = int(txt_data[line.split(":")[0]])
    # random data
    # random_state = random.randint(1, 1000)
    # train_validation_rate = random.uniform(0.1, 0.2)
    # load existed data
    # random_state=txt_data["random_seed"]
    # train_validation_rate=txt_data["test_size"]

    x_train, x_test, y_train, y_test, x_val, y_val = polar_remove_load(aug, random_state, train_validation_rate, suffix)

    # random data
    # initial_learning_rate = random.uniform(0.01, 0.1)
    # decay_steps = random.randint(1000, 5000)
    # decay_rate = random.uniform(0.1, 1)
    # STEPS = random.randint(100, 200)
    bs = int(len(x_train) / STEPS)

    # # load existed data
    # initial_learning_rate = txt_data["initial_learning_rate"]
    # decay_steps = txt_data["decay_steps"]
    # decay_rate = txt_data["decay_rate"]
    # STEPS = txt_data["STEPS"]
    # bs = txt_data["batchsize"]

    # x_train, x_test, y_train, y_test, x_val, y_val = polar_remove_load(aug)

    print('train shape{0}, validation shape {1},test shape {2}'.format
          (x_train.shape, x_val.shape, x_test.shape))
    # ########residual##########################
    from keras.callbacks import EarlyStopping
    hypermodel = ResNet18(10)
    # hypermodel=None_ResnetBlock(2);
    # #print the model# ############
    hypermodel.build(input_shape=(None, 128, 128, 2))
    hypermodel.build_graph().summary()
    tf.keras.utils.plot_model(
        hypermodel.build_graph(),  # here is the trick (for now)
        to_file='model.png', dpi=96,  # saving
        show_shapes=True, show_layer_names=True,  # show shapes and layer name
        expand_nested=False  # will show nested block
    )
    # ################## learning rate scheduler
    import random

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        # initial_learning_rate,
        initial_learning_rate=initial_learning_rate,
        # initial_learning_rate=0.01,
        decay_steps=decay_steps,
        decay_rate=decay_rate,
        # decay_steps=3000,
        # decay_rate=0.5,

        # another approach:
        # initial_learning_rate=0.01,
        # decay_steps=4000,
        # decay_rate=0.4,
        # under ~100 epochs
        #
        # (best:0.92)
        # initial_learning_rate=0.001,
        # decay_steps=3400,
        # decay_rate=0.3,
        staircase=True)
    # #########################
    hypermodel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        # optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"])
    # hypermodel.summary()
    # ############################# call back: the early stopping
    es = EarlyStopping(patience=20, restore_best_weights=True, monitor="val_accuracy")
    from keras.callbacks import ReduceLROnPlateau
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.1,
    #     patience=10,
    #     min_lr=1e-6,
    #     verbose=1)
    # ####check point path for model loading

    from event_stream import find_path
    root_dir, train_path, test_path = find_path()
    checkpoint_path = root_dir + "/dvs_SAVE_new_2/" + model_path
    import os
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    # if aug:
    #     # bs = 170
    #     bs = 340
    # else:
    #     bs = 45
    # STEPS = len(train_n) / bs
    # STEPS = 170

    history = hypermodel.fit(x_train, y_train, batch_size=bs, steps_per_epoch=STEPS,
                             # epochs=140,
                             epochs=120,
                             validation_data=(x_val, y_val),
                             callbacks=[
                                 es,
                                 cp_callback
                             ]
                             )
    # # Create a callback that saves the model's weights###########
    hypermodel.summary()
    # plot loss during training
    from datetime import datetime
    import random
    import string

    plt_loss_acc(history, name, dir_name)
    accuracy = hypermodel.evaluate(x_test, y_test, verbose=0)[1]
    print('Accuracy:', accuracy)
    # with open("max_Accuracy.txt", "r") as file:
    #     maxVal=float(file.read())
    # ####################################
    y_pred = np.array(list(map(lambda x: np.argmax(x), hypermodel.predict(x_test))))
    # ###############
    # hypermodel.load_weights("/training_1/cp.ckpt")
    # accuracy = hypermodel.evaluate(x_test, y_test, verbose=0)[1]
    # y_pred = hypermodel.predict(x_test)

    # ###########
    # if maxVal<accuracy:
    #     plt_loss_acc(history)
    #     plot_cm(y_test, y_pred, accuracy)
    #     with open("max_Accuracy.txt", "w") as file:
    #         file.write(str(accuracy))
    #     import os

    # 手动复制文件
    #         with open(checkpoint_path, "rb") as source:
    #             with open("destination_file.h5", "wb") as destination:
    #                 destination.write(source.read())

    # sum_n_acc(y_pred, y_test, n_num)

    plot_cm(y_test, y_pred, accuracy, name, dir_name)

    savepara = {"initial_learning_rate": initial_learning_rate, "decay_steps": decay_steps,
                "decay_rate": decay_rate, "STEPS": STEPS, "batchsize": bs, "accuracy": accuracy
        , "random_seed": random_state, "test_size": train_validation_rate, "tune": tune}
    with open(f"{dir_name}/{name}.txt", "w") as file:
        file.write(",\n".join([f"{name}: {value}" for name, value in savepara.items()]))
    hypermodel.save_weights(f"{dir_name}/{name}.h5")
    end = time.time()
    print('Training time: ', end - start)

    return x_train, x_val, x_test, y_train, y_val, y_test


def plot_cm(y_test, y_pred, accuracy, name, results):
    import matplotlib.pyplot as plt
    import seaborn as sns
    axis_labels = ['Clapping hands', 'Right hand\nwave', 'Left hand\nwave',
                   'Right arm\nclockwise', 'Right arm\ncounter\n-clockwise',
                   'Left arm\nclockwise', 'Left arm\ncounter\n-clockwise',
                   'Arm roll', 'Air drums', 'Air guitar']
    axis_labels = list(range(1, 11))
    # axis_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    from sklearn.metrics import confusion_matrix, classification_report

    cm = confusion_matrix(y_test, y_pred)
    cm_normal = np.round(cm / np.sum(cm, axis=1).reshape(-1, 1), 2)

    classification_report(y_test, y_pred)

    plt.figure(figsize=(12.5, 13.5))
    TITLE_FONT_SIZE = {"size": "22"}
    LABEL_SIZE = 30
    sns.set(font_scale=2.2)
    g = sns.heatmap(cm_normal, annot=True, fmt='g', cbar=False, cmap='Blues')
    # g = sns.heatmap(cm, annot=True, vmin=10, fmt='g', cbar=False, cmap='Blues'
    #                 )
    g.set_xticks(range(10))
    g.set_yticks(range(10))
    # <--- set the ticks first

    g.set_xticklabels(axis_labels)
    g.set_yticklabels(axis_labels)
    g.tick_params(axis="both", which="major", labelsize=LABEL_SIZE)
    plt.setp(g.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.setp(g.get_yticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    # plt.colorbar()
    plt.xlabel("Predicted", fontdict=TITLE_FONT_SIZE, fontweight='bold')
    plt.ylabel("Actual", fontdict=TITLE_FONT_SIZE, fontweight='bold')
    plt.title("Confusion Matrix", fontdict=TITLE_FONT_SIZE, fontweight='bold')
    # plt.colorbar(g)
    g.text(40, -2.4, 'Accuracy {0}'.format(accuracy * 100), fontsize=14, color='black')

    plt.savefig(f'{results}/{name}_cm.png')
    plt.show()
    plt.clf()
    # g.clf()


def plt_loss_acc(history, name, results):
    # # ###########
    # plot loss during training
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    linewidth = '2'
    axs[0].set_title('Loss / Mean Squared Error')
    axs[0].plot(history.history['loss'], label='Training', linewidth=linewidth)
    axs[0].plot(history.history['val_loss'], label='Validation', linewidth=linewidth)
    axs[0].set(ylabel='Loss')
    axs[0].legend(loc='lower right')

    axs[1].set_title('Epoch-Accuracy')
    axs[1].plot(history.history['accuracy'], label='Accuracy', linewidth=linewidth)
    axs[1].plot(history.history['val_accuracy'], label='Val_Accuracy', linewidth=linewidth)
    axs[1].set(xlabel='Epoch', ylabel='Accuracy')
    axs[1].legend(loc='lower right')

    fig.tight_layout()
    fig.savefig(f'{results}/{name}_loss.png')
    fig.show()


def dataset_generator_and_training(para_before_tune, para_after_tune, tune_choice: list, suffix: str,
                                   if_only_train=True):
    root_dir, train_save_path, test_save_path = find_path(suffix)
    import os
    results_save_path = os.path.join(test_save_path, "test_results")
    if if_only_train:
        os.makedirs(results_save_path, exist_ok=True)
        for i in range(100):
            hyper_tuner_for_times(True, False, "results", 1, results_save_path, suffix)

    polarity_process_transistor_conditions(True, para_before_tune, para_after_tune, suffix, tune_choice)
    polarity_process_transistor_conditions(False, para_before_tune, para_after_tune, suffix, tune_choice)
    params = vars(para_after_tune)
    import os
    save_params_to_file(os.path.join(train_save_path, "params.txt"), **params)
    save_params_to_file(os.path.join(test_save_path, "params.txt"), **params)
    gen_augmentation_frame(suffix)
    polar_remove_set7(True, suffix)
    results_save_path = os.path.join(test_save_path, "test_results")
    os.makedirs(results_save_path, exist_ok=True)
    for i in range(100):
        hyper_tuner_for_times(True, False, "results", 1, results_save_path, suffix)
