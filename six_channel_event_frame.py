import math

from event_stream import index_arr
from params_adjustment import init_event, find_path

c = 3E-4

# c = 1E-5
# the pulse interval (in pn vs id curve)
t_interval = 2E-4

# 1. n_clip: decide the length of each part (split the 6-second event stream into n_clip shorter event stream)
# (e.g. 2s event stream => n_clip = 3)
# n_clip = 3

n_clip = 3
# 2. n_num: sum of divided parts(5=>split into 5 parts)!!!!!!!when change the n_num, the n_step may need to reset
# n_num = 5

# n_num=3

n_num = 3
# 3. n_step: the step (the gap of the events number from the beginning of present
# part to the beginning of next part) (e.g. 6=>total events/6 as step length)

# n_step=6
n_step = 6
def find_idx(t0,t):
    for i,v in enumerate(t):
        if v >= t0:
            return i
    return len(t)-1

def polar_save_as_list(set_eve, n, indexarr):
    event, label = set_eve[n]
    x0 = tuple(event["x"])
    y0 = tuple(event["y"])
    t = tuple(event["t"])
    p = tuple(event["p"])
    t=[(t0-t[0])*1e-6*c for t0 in t]
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

    t_whole=6 * c
    t_step=t_whole/n_step
    t_clip=t_whole/n_clip
    for j in range(n_num):
        t_begin=j*t_step
        t_end=t_begin+t_clip
        if t_begin>t[-1]:
            for h in range(128 * 128):
                events_dict_pos[h] = []
                events_dict_neg[h] = []
                events_dict_pos_time[h] = []
                events_dict_neg_time[h] = []
            dict_pos_time[j] = dict(events_dict_pos_time)
            dict_neg_time[j] = dict(events_dict_neg_time)


        # n = find_idx(t_begin,t)
        #
        # ne = find_idx(t_end,t)
        # print('n is {}'.format(n))
        # print('ne is {}'.format(ne))
        end_t = t_end-t_begin
        for h in range(128 * 128):
            events_dict_pos[h] = []
            events_dict_neg[h] = []
            events_dict_pos_time[h] = []
            events_dict_neg_time[h] = []
            # print(len(t))
        for i in range(len(t)):
            key = indexarr[x0[i]][y0[i]]
            if t_end > t[i] > t_begin:
                if p[i] != 1:
                    events_dict_neg[key].append((t[i] - t_begin))
                else:
                    events_dict_pos[key].append((t[i] - t_begin))

        end_time.append(end_t)

        dict_pos[j] = dict(events_dict_pos)
        dict_neg[j] = dict(events_dict_neg)
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
        dict_pos_time[j] = dict(events_dict_pos_time)
        dict_neg_time[j] = dict(events_dict_neg_time)

    # return dict_pos, dict_neg, label, end_time
    return dict_pos_time, dict_neg_time, label

def polarity_process_transistor_conditions(train: bool, para, suffix: str):
    import time
    begin = time.time()
    indexarr = index_arr()
    train_set_eve, test_set_eve = init_event(suffix=suffix)
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

    for n in tqdm(range(data_num), desc="process"):
        event, label = set_eve[n]
        if label != 2:
            i_d = para
            output_arr = np.empty((n_num, 128, 128, 2))
            dict_pos_time, dict_neg_time, label = polar_save_as_list(set_eve, n, indexarr)
            dict_list = [dict_pos_time, dict_neg_time]
            label_arr.append(label)
            # each class, generate 30 frames
            # for d in range(30):
            for d in range(n_num):
                if d not in dict_pos_time:
                    break


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

                                id_b, id_a = id_time_new(id_last, j, y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_, l_a, l_b,30.5)

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
                            output_arr[d,k, m, polar] = temp_save[polar][int(indexarr[m][k])]
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

def id_time_new(i_last, t, y0, A1, A2, A3, t1, t2, t3, d, l_a, l_b,i_l=30.5):
    id_b = (i_last / d) * (A1 * math.exp(-t / t1) + A2 * math.exp(-t / t2) + A3 * math.exp(-t / t3) + y0)
    # print(id_vs_time)
    id_a = l_a * id_b + l_b
    if id_a > i_l:
        id_a = i_l
    return id_b, id_a

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
    import numpy as np
    x_rota1 = RandomRotation(factor=(-0.1, 0), fill_mode='reflect', interpolation='nearest')
    x_rota2 = RandomRotation(factor=(0, 0.1), fill_mode='reflect', interpolation='nearest')
    x_shift = RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1),
                                fill_mode='reflect', fill_value=0.0, interpolation='nearest')
    x_shift_1 = Sequential([
        RandomRotation(factor=(-0.1, 0), fill_mode='constant', interpolation='nearest'),
        RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1),
                          fill_mode='constant', fill_value=0.0, interpolation='nearest')])
    x_shift_2 = Sequential([
        RandomRotation(factor=(0, 0.1), fill_mode='constant', interpolation='nearest'),
        RandomTranslation(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1),
                          fill_mode='constant', fill_value=0.0, interpolation='nearest')])
    y = empty(6)
    y.fill(int(y_train))
    x_aug = [[] for _ in range(6)]
    for x_clip in x_train:
        x_aug[0].append(x_clip)
        x_aug[1].append(x_rota1(x_clip))
        x_aug[2].append(x_rota2(x_clip))
        x_aug[3].append(x_shift(x_clip))
        x_aug[4].append(x_shift_1(x_clip))
        x_aug[5].append(x_shift_2(x_clip))
    # y_list = row_stack((y_list, y))
    x=np.array(x_aug)
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
            if x.ndim < 5:
                x = expand_dims(x, axis=0)
                print("expanded for x")
            y_1 = np.full(a, y[i])

            # stack (creating a initial array)
            if i != 0:
                x_0[i * a:(i + 1) * a,:,:, :, :] = x
                y_0[i * a:(i + 1) * a, ] = y_1
                # print(y_0.shape)
            else:
                x_0 = np.zeros((len(y) * a,n_num,128, 128, 2))
                y_0 = np.zeros((len(y) * a,))
                x_0[0:a, :, :, :] = x
                y_0[0:a, ] = y_1
            # os.remove(path)
        print('final x shape: {0}, and len_y is {1}'.format(x_0.shape, len(y_0)))

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