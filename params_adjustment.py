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


######CONFIGURATION################
# Path #####################
def find_path(suffix:str):
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
def init_event(spikingjelly=None):
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture as D128G
    root_dir, train_save_path, test_save_path = find_path()
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
def polarity_process_transistor_conditions(train: bool, y0_off=None, a1_off=None, t1_off=None, a2_off=None, t2_off=None,
                                           a3_off=None, t3_off=None,
                                           d_off=None, a_off=None, b_off=None):
    import time
    begin = time.time()
    indexarr = index_arr()
    train_set_eve, test_set_eve = init_event()
    root_dir, train_save_path, test_save_path = find_path()
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
    i_d = calculate_match(y0_off, a1_off, t1_off, a2_off, t2_off,
                          a3_off, t3_off,
                          d_off, a_off, b_off)

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
    def __init__(self, y0_off=None, a1_off=None, t1_off=None, a2_off=None, t2_off=None, a3_off=None, t3_off=None,
                 d_off=None, a_off=None, b_off=None):

        if y0_off is None:
            y0_off = [0, 0, 0]
        if a1_off is None:
            a1_off = [0, 0, 0]
        if t1_off is None:
            t1_off = [0, 0, 0]
        if a2_off is None:
            a2_off = [0, 0, 0]
        if t2_off is None:
            t2_off = [0, 0, 0]
        if a3_off is None:
            a3_off = [0, 0, 0]
        if t3_off is None:
            t3_off = [0, 0, 0]
        if d_off is None:
            d_off = [0, 0, 0]
        if a_off is None:
            a_off = 0
        if b_off is None:
            b_off = 0

        self.y0 = [20.06402, 20.31625, 20.55575]
        self.A1 = [6.30373, 6.65462, 9.10203]
        self.t1 = [1.49676E-5, 1.93912E-5, 2.84333E-5]
        self.A2 = [4.72692, 4.61784, 2.19681]
        self.t2 = [8.72838E-5, 1.09493E-4, 2.77744E-4]
        self.A3 = [0.5666, 0.45408, 0.59678]
        self.t3 = [0.01394, 0.12556, 1.29371]

        self.y0 = [a + b for a, b in zip(self.y0, y0_off)]
        self.A1 = [a + b for a, b in zip(self.A1, a1_off)]
        self.y0 = [a + b for a, b in zip(self.t1, t1_off)]
        self.y0 = [a + b for a, b in zip(self.A2, a2_off)]
        self.y0 = [a + b for a, b in zip(self.t2, t2_off)]
        self.y0 = [a + b for a, b in zip(self.A3, a3_off)]
        self.y0 = [a + b for a, b in zip(self.t3, t3_off)]
        # self.y0 = [19.91713, 20.34551, 20.67151]
        # self.A1 = [7.52151, 7.61645, 8.42568]
        # self.A2 = [2.72444, 3.31655, 2.87707]
        # self.A3 = [0.34774, 0.42499, 0.54843]
        # self.t1 = [2.56303E-5, 2.3492E-5, 2.46777E-5]
        # self.t2 = [1.70132E-4, 1.50927E-4, 1.81061E-4]
        # self.t3 = [0.39407, 0.10214, 0.47215]

        # d: threshold for shifting the relaxation formula
        self.d = [28.81179, 29.65465, 30.59423]
        self.d = [a + b for a, b in zip(self.d, d_off)]

        self.a = 0.89091
        self.a = self.a + a_off

        self.b = 6.78201
        self.b = self.b + b_off

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
    # if id_a > 30.5:
    #     id_a = 30.5
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


def gen_augmentation_frame():
    import os
    import numpy as np
    import time
    from event_stream import find_path
    begin = time.time()
    root_dir, train_path, test_path = find_path()
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
    gen_stack_frame(Aug=True)
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


def gen_stack_frame(Aug: bool):
    import os
    import numpy as np
    import time
    from event_stream import find_path
    begin = time.time()
    root_dir, train_path, test_path = find_path()
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

def dataset_generator(y0_off=None, a1_off=None, t1_off=None, a2_off=None, t2_off=None,
                                           a3_off=None, t3_off=None,
                                           d_off=None, a_off=None, b_off=None):
    polarity_process_transistor_conditions(True,y0_off, a1_off, t1_off, a2_off, t2_off,
                          a3_off, t3_off,
                          d_off, a_off, b_off)
    polarity_process_transistor_conditions(False,y0_off, a1_off, t1_off, a2_off, t2_off,
                          a3_off, t3_off,
                          d_off, a_off, b_off)
    params=locals()
    root_dir, train_save_path, test_save_path=find_path("eve_1")
    import os
    save_params_to_file(os.path.join(train_save_path,"params.txt"),**params)
    save_params_to_file(os.path.join(test_save_path,"params.txt"),**params)
    gen_augmentation_frame()


