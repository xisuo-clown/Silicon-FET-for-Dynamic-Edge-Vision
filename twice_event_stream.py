def init_event(spikingjelly=None):
    from spikingjelly.datasets.dvs128_gesture import DVS128Gesture as D128G
    root_dir, train_save_path, test_save_path = find_path()
    train_set_eve = D128G(root_dir, train=True, data_type='event')
    test_set_eve = D128G(root_dir, train=False, data_type='event')
    return train_set_eve, test_set_eve


def find_path(prefix: str):
    # Original event stream file path
    import os
    root_dir = os.path.join(os.getcwd(), "DvsGes")
    # path for event array saving: train and test
    train_save_path = os.path.join(root_dir, prefix,"train_set_eve")
    test_save_path = os.path.join(root_dir, prefix,"test_set_eve")
    # train_save_path = (root_dir + '/CBRAM/train_set_eve/')
    # test_save_path = (root_dir + "/CBRAM/test_set_eve/")
    return root_dir, train_save_path, test_save_path


def index_arr():
    import numpy as np
    h = 0
    indexarr = np.zeros((128, 128))
    for i in range(128):
        for j in range(128):
            indexarr[i][j] = h
            h += 1
    return indexarr


def polarity_process_transistor_twice_conditions(train: bool):
    import time
    begin = time.time()
    indexarr = index_arr()
    train_set_eve, test_set_eve = init_event()
    root_dir, train_save_path, test_save_path = find_path("event_array_twice_transistor")
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
