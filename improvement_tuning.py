#
# Parameters ################
# coefficient: rescal time axis  e.g. 1s => c*1s
# rescale c.
# c = 1E-4##############
tuner_c = 3E-4
# tuner_c = 0.6E-3
# the pulse interval (in pn vs id curve) required when using 'match model'
from event_stream import t_interval

# 1. n_clip: decide the length of each part
# (e.g. for total 6s event stream, each part contains 2s data => n_clip = 6/2 =3)
# !!!!!when change the clip length, the memory length tuner_n_clip should be simultaneously modified!
# tuner_n_clip = 6
# tuner_n_clip = 4
tuner_n_clip = 3
from event_stream import n_num, n_step
import math
import numba as nb


# 3. n_step: the step (the gap of the events number from the beginning of present
# part to the beginning of next part) (e.g. 6=>[total events/6] as step length)
# tuner_n_step = 6


################relaxation and potentiation#####################
def para_transistor_exp():
    # time(-1.4v, 10ms)
    # a1, a2, tau1, tau2 = 2.54079, 0.58651, 2.58652E-5, 5.46109E-4
    a1, a2, tau1, tau2 = 1.98088E-4, 0.04903, 8.41192E-7, 1.98484
    # pulse number(-1.4v, 200us)
    # unit: uA
    t1, y1 = -57.98876, 9.46142E-5


    i_first = 0.00000259
    return a1, a2, tau1, tau2, t1, y1, i_first


# ###################parameters setting##############
class calculate_match_tune:
    def __init__(self):
        self.y0 = [20.01978, 20.42166, 20.75204]
        self.A1 = [5.15863, 7.37592, 7.96576]
        self.t1 = [7.53665E-5, 2.2508E-5, 2.60293E-5]
        self.A2 = [5.13078, 2.82181, 2.64961]
        self.t2 = [1.32347E-5, 1.52147E-4, 1.96754E-4]
        self.A3 = [0.35981, 0.35254, 0.38272]
        self.t3 = [0.02205, 0.31306, 0.25639]

        self.d = [27.9274791, 28.8670617, 29.73755545]

        # # for 500us
        # self.a = 0.60868
        # self.b = 13.56075

        # for 100us
        self.a = 0.68193
        self.b = 11.48977


        # id after first pulse
        self.id_0 = self.d[0]
        # the maximum of id
        self.id_th = self.d[2]
        # threshold for shifting relaxation curve
        # !!!!!!!!!!!!!!!!!!!
        #

        # self.i_1 = self.d[1]
        # self.i_2 = self.d[2]


        # self.y0 = [20.01978, 20.37088, 20.75204]
        # self.A1 = [5.15863, 6.35407, 7.96576]
        # self.A2 = [5.13078, 3.68864, 2.64961]
        # self.A3 = [0.35981, 0.36548, 0.38272]
        # self.t1 = [7.53665E-5, 2.06681E-5, 2.60293E-5]
        # self.t2 = [1.32347E-5, 1.11163E-4, 1.96754E-4]
        # self.t3 = [0.02205, 0.03312, 0.25639]
        # self.d = [28.67, 29.54, 30.44]
        # self.a = 0.681
        # self.b = 11.489
        # # id after first pulse
        # self.id_0 = 28.2
        # # the maximum of id
        # self.id_th = 29.5
        # # threshold for shifting relaxation curve
        # # !!!!!!!!!!!!!!!!!!!
        # #
        # self.i_1 = 29.2
        # self.i_2 = 28.4

    def get_para(self, i_last):
        if i_last < self.d[1]:
            i = 0
        elif i_last < self.d[2]:
            i = 1
        else:
            i = 2
        l_a, l_b = self.a, self.b
        y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_ = (self.y0[i], self.A1[i], self.A2[i],
                                                 self.A3[i], self.t1[i], self.t2[i],
                                                 self.t3[i], self.d[i])
        return y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_, l_a, l_b


############################################################

# Generate the frames (in class n) under another pulse config
def add_polarity_process_transistor(n, train: bool, calculate: bool):
    import time
    begin = time.time()
    from event_stream import (index_arr, init_event, find_path, id_time, id_num_exp, n_num)
    indexarr = index_arr()
    if n > 1:
        n_1 = n + 1
    else:
        n_1 = n
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

    from tqdm import tqdm
    import numpy as np
    # n_list = find_set_event(n_1, data_num, set_eve)
    # print(n_list)
    n_list = []
    for i in range(data_num):
        event, label = set_eve[i]
        if int(label) == n_1:
            n_list.append(i)

    if calculate:
        process_cal(n, n_list, label_arr, set_eve, indexarr, save_path)
    else:
        process_match(n, n_list, label_arr, set_eve, indexarr, save_path)

    end = time.time()
    print(end - begin)
    # Events stream transform into array in n.npy, print arrays to visualize frames
    return


def process_cal(n, n_list, label_arr, set_eve, indexarr, save_path):
    import numpy as np
    from tqdm import tqdm
    from event_stream import (id_time, id_num_exp, n_num)
    a1, a2, tau1, tau2, t1, y1, i_first = para_transistor_exp()
    for l in tqdm(n_list, desc="process"):
        output_arr = np.empty((128, 128, 2))
        temp_save = []
        dict_pos_time, dict_neg_time, label = add_polar_save_as_list(set_eve, l, indexarr)
        dict_list = [dict_pos_time, dict_neg_time]
        # here we removed class 2 (other gestures)
        # each class, generate 30 frames
        # for d in range(30):
        for d in range(n_num):
            label_arr.append(l * n_num + d - 1)
            for q in range(2):
                events_dict_time = dict_list[q][d]
                # print(events_dict)
                # print(end_time[d])
                for i in range(128 * 128):
                    if events_dict_time[i]:
                        i_last = i_first
                        #####instant current value calculation for each pixel(transistor)
                        for j in range(len(events_dict_time[i]) - 1):
                            t = events_dict_time[i][j]
                            id_b_p = id_time(i_last, t, a1, a2, tau1, tau2)
                            # print(id_b_p)
                            i_last = id_num_exp(id_b_p, t1, y1)
                        #####
                        t = events_dict_time[i][len(events_dict_time[i]) - 1]
                        # print("t is {}".format(t))
                        id_b_p = id_time(i_last, t, a1, a2, tau1, tau2)
                        # print(id_b_p)
                        temp_save.append(id_b_p)
                    else:
                        temp_save.append(0)
                for k in range(128):
                    for m in range(128):
                        output_arr[k, m, q] = temp_save[int(indexarr[m][k])]
                np.save(save_path + "class{0}_tuned_{1}.npy".format(n, l * n_num + d - 1), output_arr)
    np.save(save_path + "class{0}_tuned_dataset_index.npy".format(n), label_arr)
    y = np.load(save_path + "dataset_labels.npy")
    return y


def process_match(n, n_list, label_arr, set_eve, indexarr, save_path):
    import numpy as np
    from tqdm import tqdm
    from event_stream import (id_time, take_closest, n_num, id_time_new)
    from statistics import mean
    # # ##########old version begin#####################
    # from event_stream import generate_comparison_idpd_index, para_transistor_bi_exp
    # a1, a2, tau1, tau2, b1, b2, t1, t2, y1 = para_transistor_bi_exp()
    # id_list, id_pulse_dict = generate_comparison_idpd_index()
    # # ##########old version end##################

    # ##########new version begin#####################
    i_d = calculate_match_tune()
    # ##########new version end##################

    for l in tqdm(n_list, desc="process"):
        output_arr = np.empty((128, 128, 2))
        temp_save = []
        dict_pos_time, dict_neg_time, label = add_polar_save_as_list(set_eve, l, indexarr)
        dict_list = [dict_pos_time, dict_neg_time]
        # here we removed class 2 (other gestures)
        # each class, generate 30 frames
        # for d in range(30):
        for d in range(n_num):
            label_arr.append(l * n_num + d - 1)
            for q in range(2):
                events_dict_time = dict_list[q][d]
                # print(events_dict)
                # print(end_time[d])
                for i in range(128 * 128):
                    if events_dict_time[i]:
                        id_last = i_d.id_0
                        y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_, l_a, l_b = i_d.get_para(id_last)
                        for j in events_dict_time[i]:
                            # ###########new version begin###################
                            id_b_p, id_last = id_time_tune_new(id_last, j, y_0, A_1, A_2, A_3,
                                                               t_1, t_2, t_3, d_, l_a, l_b, i_d.id_th, i_d.id_0)
                        # ###########new version end####################################
                        # # @@############old version begin###
                        # for j in range(len(events_dict_time[i]) - 1):
                        #     i_last = id_time(i_last, events_dict_time[i][j], a1, a2, tau1, tau2)
                        #     # print("the present is{}".format(i_last))
                        #     i_last = take_closest(id_list, i_last, id_pulse_dict)
                        #     # print("the closest is {}".format(i_last))
                        # # print('last t is {}'.format(t))
                        # id_b_p = id_time(i_last, events_dict_time[i][-1], a1, a2, tau1, tau2)
                        # # print('last id_b_p is {}'.format(id_b_p))
                        # # @@##########old version end#######################################

                        if id_b_p - 15.2 > 0:
                            temp_save.append(id_b_p - 15.2)
                        else:
                            temp_save.append(0)
                        # temp_save.append(id_b_p)
                    else:
                        temp_save.append(0)
                for k in range(128):
                    for m in range(128):
                        output_arr[k, m, q] = temp_save[int(indexarr[m][k])]
                # print("the mean value is {}".format(mean(temp_save)))
                np.save(save_path + "class{0}_tuned_{1}.npy".format(n, l * n_num + d - 1), output_arr)
    np.save(save_path + "class{0}_tuned_dataset_index.npy".format(n), label_arr)


# just augment the newly generated frame *same augment as previous
# encode the labels
def gen_augmentation_frame(n):
    import os
    import numpy as np
    import time
    from event_stream import find_path
    begin = time.time()
    root_dir, train_path, test_path = find_path()
    # processing
    import tensorflow as tf
    from numpy import expand_dims
    from tqdm import tqdm
    from frames_processing import aug_process
    # ####################
    path_type = [train_path, test_path]
    name_type = ['Train', 'Test']
    name_save = ['Aug_', '']

    # # # # # save the aug data#########################################
    tuned_y = np.load(os.path.join(train_path, "class{0}_tuned_dataset_index.npy".format(n)), allow_pickle=True)
    tuned_y = tuned_y.tolist()
    for i in tqdm(tuned_y, desc="Rotate and shift-Train"):
        path = os.path.join(train_path, "class{0}_tuned_{1}.npy".format(n, i))
        x_train = np.load(path, allow_pickle=True)
        x, y = aug_process(x_train, n)
        np.save(train_path + "Aug_class{0}_tuned_{1}.npy".format(n, i), x)
    # # # # # $$$$$$$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%############
    # #
    # # stack with other classes
    for j in range(2):
        tuned_y = np.load(os.path.join(path_type[j], "class{0}_tuned_dataset_index.npy".format(n)), allow_pickle=True)
        # tuned_y = np.load(os.path.join(path_type[j], "class{0}_tuned_dataset_labels.npy".format(n)), allow_pickle=True)
        print(tuned_y.shape)
        # ######################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        tuned_y = tuned_y.tolist()
        for i in tqdm(range(len(tuned_y)), desc="Stack data"):
            path = os.path.join(path_type[j], "{0}class{1}_tuned_{2}.npy".format(name_save[j], n, tuned_y[i]))
            x = np.load(path, allow_pickle=True)
            if j == 1:
                x = expand_dims(x, axis=0)
            if i == 0:
                x_0 = x
            else:
                x_0 = np.row_stack((x_0, x))
            # os.remove(os.path.join(path_type[j], "{0}class{1}_tuned_{2}.npy".format(name_save[j], n, tuned_y[i])))
        np.save(path_type[j] + "{0}class{1}_tuned_add.npy".format(name_save[j], n), x_0)
    print('Stack finish')

    # split origin augmented data
    for j in range(2):
        # for j in [1]:
        y = np.load(os.path.join(path_type[j], "{}dataset_labels.npy".format(name_save[j])), allow_pickle=True)
        x = np.load(os.path.join(path_type[j], "{}dataset_features.npy".format(name_save[j])), allow_pickle=True)
        print("label length {}".format(len(y)))
        from sklearn.preprocessing import LabelEncoder
        y_labelencoder = LabelEncoder()
        y = y_labelencoder.fit_transform(y)
        y_i = y.tolist()
        for i in range(len(y_i)):
            if y_i[i] == n:
                b_0 = i
                break
        for i in range(len(y_i)):
            if y_i[i] == n:
                b_1 = i
        print("{0}: n begins from {1}, ends at {2}".format(name_type[j], b_0, b_1))
        x_1 = x[:b_0, :, :, :]
        np.save(os.path.join(path_type[j], "{}dataset_features_1a.npy".format(name_save[j])), x_1)
        y_1 = y[:b_0, ]
        np.save(os.path.join(path_type[j], "{}dataset_labels_1a.npy".format(name_save[j])), y_1)
        x_2 = x[b_1 + 1:, :, :, :]
        np.save(os.path.join(path_type[j], "{}dataset_features_2a.npy".format(name_save[j])), x_2)
        y_2 = y[b_1 + 1:, ]
        np.save(os.path.join(path_type[j], "{}dataset_labels_2a.npy".format(name_save[j])), y_2)
    print('Split finish')

    # combine
    for j in range(2):
        x_0 = np.load(path_type[j] + "{0}class{1}_tuned_add.npy".format(name_save[j], n), allow_pickle=True)
        y_0 = np.empty(x_0.shape[0])
        y_0.fill(n)
        print('the shape is {0}'.format(x_0.shape))
        x_1 = np.load(os.path.join(path_type[j], "{}dataset_features_1a.npy".format(name_save[j])), allow_pickle=True)
        x = np.concatenate([x_1, x_0], axis=0)
        x_2 = np.load(os.path.join(path_type[j], "{}dataset_features_2a.npy".format(name_save[j])), allow_pickle=True)
        x = np.concatenate([x, x_2], axis=0)
        np.save(os.path.join(path_type[j], "tuned_{}dataset_features.npy".format(name_save[j])), x)

        y_1 = np.load(os.path.join(path_type[j], "{}dataset_labels_1a.npy".format(name_save[j])), allow_pickle=True)
        y = np.concatenate([y_1, y_0], axis=0)
        y_2 = np.load(os.path.join(path_type[j], "{}dataset_labels_2a.npy".format(name_save[j])), allow_pickle=True)
        y = np.concatenate([y, y_2], axis=0)
        np.save(os.path.join(path_type[j], "tuned_{}dataset_labels.npy".format(name_save[j])), y)
        print(x.shape)
        print(y.shape)

        os.remove((path_type[j] + "{}dataset_features_1a.npy".format(name_save[j])))
        os.remove((path_type[j] + "{}dataset_features_2a.npy".format(name_save[j])))
        os.remove((path_type[j] + "{}dataset_labels_1a.npy".format(name_save[j])))
        os.remove((path_type[j] + "{}dataset_labels_2a.npy".format(name_save[j])))
    print('combine finish')


# without augmentation
def gen_tuned_stack_frame(n, tune: bool):
    name_type = ['Train', 'Test']
    name_save = ['Ori_', '']
    import time
    from event_stream import find_path
    begin = time.time()
    root_dir, train_path, test_path = find_path()
    path_type = [train_path, test_path]
    from numpy import expand_dims
    import os
    import numpy as np
    for j in range(2):
        y = np.load(os.path.join(path_type[j], "class{0}_tuned_dataset_index.npy".format(n)), allow_pickle=True)
        y = y.tolist()
        tuned_list = []
        for h in y:
            x = np.load(os.path.join(path_type[j], "class{0}_tuned_{1}.npy".format(n, h)), allow_pickle=True)
            try:
                x_0 = np.row_stack((x_0, expand_dims(x, axis=0)))
            except:
                x_0 = expand_dims(x, axis=0)
            tuned_list.append(n)
        print(x_0.shape)
        if tune:
            ori_x = np.load(path_type[j] + "{}dataset_features.npy".format(name_save[j]), allow_pickle=True)
            ori_y = np.load(path_type[j] + "{}dataset_labels.npy".format(name_save[j]), allow_pickle=True)

        else:
            ori_x = np.load(path_type[j] + "tuned_{}dataset_features.npy".format(name_save[j]), allow_pickle=True)
            ori_y = np.load(path_type[j] + "tuned_{}dataset_labels.npy".format(name_save[j]), allow_pickle=True)

        print(ori_y.shape)
        print(ori_x.shape)
        ori_y = ori_y.tolist()
        ori_list = []
        # should rewrite( create a new list contains tuned labels
        for h in range(len(ori_y)):
            if ori_y[h] == n:
                ori_list.append(h)
        a = int(ori_list[0])

        tuned_y = np.concatenate([ori_y[:ori_list[0], :, :, :],
                                  ori_y[ori_list[len(ori_list)]:, :, :, :]],
                                 axis=0)
        #     test
        for h in range(len(tuned_y)):
            if tuned_y[h] == n:
                print('error!')
        tuned_x = np.concatenate([ori_x[:ori_list[0], :, :, :], x_0,
                                  ori_x[ori_list[len(ori_list)]:, :, :, :]], axis=0)
        np.save(path_type[j] + "tuned_{}dataset_features.npy".format(name_save[j]), tuned_x)
        np.save(path_type[j] + "tuned_{}dataset_labels.npy".format(name_save[j]), tuned_y)
    print('END: Stacking')
    end = time.time()
    print(end - begin)


def load_tuned_removed7(Aug: bool,random_state=42,test_size=0.125):
    import time
    from event_stream import find_path
    import numpy as np
    begin = time.time()
    root_dir, train_path, test_path = find_path()
    path_type = [train_path, test_path]
    input_name = ['Aug_', 'Ori_']

    x_train, x_test = np.array([]), np.array([])
    x = [x_train, x_test]
    y_train, y_test = np.array([]), np.array([])
    y = [y_train, y_test]
    # np.save(path_type[j] + "tuned_{}dataset_features.npy".format(name_save[j]), tuned_x)
    # np.save(path_type[j] + "tuned_{}dataset_labels.npy".format(name_save[j]), tuned_y)
    if Aug:
        name_save = [input_name[0], '']
    else:
        name_save = [input_name[1], '']

    for j in range(2):
        x[j] = np.load(path_type[j] + "tuned_{}dataset_features.npy".format(name_save[j]), allow_pickle=True)
        y[j] = np.load(path_type[j] + "tuned_{}dataset_labels.npy".format(name_save[j]), allow_pickle=True)
        b_6, e_6, b_7, e_7 = cal_set_7(y[j])
        # x[j] = np.concatenate([x[j][:b_7, :, :, :], x[j][b_7+e_6-b_6+1:, :, :, :]], axis=0)
        # y[j] = np.concatenate([y[j][:b_7, ], y[j][b_7+e_6-b_6+1:, ]], axis=0)
        x[j] = np.concatenate([x[j][:e_7 - e_6 + b_6, :, :, :], x[j][e_7 + 1:, :, :, :]], axis=0)
        y[j] = np.concatenate([y[j][:e_7 - e_6 + b_6, ], y[j][e_7 + 1:, ]], axis=0)
        # #######only enable in test--------begin----
        # d_6 = 0
        # d_7 = 0
        # for i in y[j]:
        #     if i == 6:
        #         d_6 += 1
        #     if i == 7:
        #         d_7 += 1
        # print('after removed set7, the number of class 6 is {0}, class 7 is {1}'.format(d_6, d_7))
        # #######only enable in test---------end----
        print(x[j].shape)
        print(y[j].shape)
    from sklearn.model_selection import train_test_split
    from frames_processing import frame_normalization
    x_train, x_test = x[0], x[1]
    y_train, y_test = y[0], y[1]
    x_train, x_test = frame_normalization(x_train, x_test)
    seed = random_state
    x_train, x_val, y_train, y_val = (
        train_test_split(x_train, y_train, test_size=test_size, random_state=seed))
    return x_train, x_test, y_train, y_test, x_val, y_val


# ################################

def add_polar_save_as_list(set_eve, class_n, indexarr):
    event, label = set_eve[class_n]
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
        ne = n + int(len(t) / tuner_n_clip) - 1
        # print('n is {}'.format(n))
        # print('ne is {}'.format(ne))
        end_t = (t[ne] - t[n]) * 1E-6 * tuner_c
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
                    events_dict_neg[key].append((t[i] - t[n]) * 1E-6 * tuner_c)
                else:
                    events_dict_pos[key].append((t[i] - t[n]) * 1E-6 * tuner_c)
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
                    for l in events_dict[k]:
                        if l != events_dict[k][0]:
                            events_dict_time[k].append(l - events_dict[k][0])
                        events_dict_time[k].append(end_t - events_dict[k][len(events_dict[k]) - 1])
        dict_pos_time[j] = events_dict_pos_time
        dict_neg_time[j] = events_dict_neg_time

    # return dict_pos, dict_neg, label, end_time
    return dict_pos_time, dict_neg_time, label


# find set n, and generate list contains the gesture numbers belongs to class n
def find_set_event(n, data_num, set_eve):
    n_list = []
    for i in range(data_num):
        event, label = set_eve[i]
        if int(label) == n:
            n_list.append(i)
    return n_list


# find the correct length
def cal_set_7(y):
    y = y.tolist()
    for i in range(len(y)):
        if y[i] == 6:
            b_6 = i
            break
    for i in range(len(y)):
        if y[i] == 6:
            e_6 = i
        if y[i] == 7:
            b_7 = i
            break
    for i in range(len(y)):
        if y[i] == 7:
            e_7 = i

    return b_6, e_6, b_7, e_7


@nb.jit(nopython=True)
def id_time_tune_new(i_last, t, y0, A1, A2, A3, t1, t2, t3, d, l_a, l_b, Id_th, Id_th_0):
    id_b = (i_last / d) * (A1 * math.exp(-t / t1) + A2 * math.exp(-t / t2) + A3 * math.exp(-t / t3) + y0)
    # print(id_vs_time)
    id_a = l_a * id_b + l_b
    if id_a > Id_th:
        id_a = Id_th
    if id_a < Id_th_0:
        id_a = Id_th_0
    return id_b, id_a
