# import libraries
import os
from event_stream import calculate_match, id_time_new

import tensorflow
import tonic
import torch
import numpy as np
import pandas as pd
import numba as nb
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict

from event_stream import generate_comparison_idpd_index, para_transistor_bi_exp, take_closest, id_time, index_arr


def Collect_Frames():
    dataPath = "C:/Users/ASUS/OneDrive - Nanyang Technological University/datasets/DVSAnimals/data/"
    dataset = AnimalsDvsSliced(dataPath)
    print("the length is :{}".format(dataset.__len__()))
    list_labels = []
    for i in tqdm(range(dataset.__len__()), desc="Event Processing"):
        # label = dataset.Polarity_Match(i)
        label = dataset.Polarity_cal(i)
        list_labels.append(int(label))
    np.save(dataPath + "events/dataset_labels.npy", list_labels)

    x = gen_train_test(dataPath=dataPath, data_num=dataset.__len__())
    x.stack_frames()
    x.stack_aug()


# check the length of each event stream
def check_length():
    dataPath = "C:/Users/ASUS/OneDrive - Nanyang Technological University/datasets/DVSAnimals/data/"
    dataset = AnimalsDvsSliced(dataPath)
    print("the length is :{}".format(dataset.__len__()))
    # i=901
    #
    # events, class_name, class_index, target = dataset.__getitem__(index=i)
    # print("events is {0},\n class name is {1},\n target is {2},\n "
    #       "class index is {3}\n INDEX {4}".format(events,class_name,target,class_index, i))
    shortest_time = 5
    time_save = defaultdict(list)
    short_time_sum = [0] * 19
    index_name = []
    for i in tqdm(range(dataset.__len__()), desc="Event Processing"):
        events, class_name, class_index = dataset.__getitem__(i)
        len_time = (events["t"][-1] - events["t"][0]) * 1E-6
        # print("events time is {0},\n class name is {1},\n "
        #       "class index is {2}\n INDEX {3}".format(len_time, class_name, class_index, i))
        time_save[class_index].append(len_time)
        if len_time < shortest_time:
            shortest_time = len_time
            class_name_0 = class_name
        if len_time < 2:
            short_time_sum[class_index] += 1
            index_name.append(i)
    print("the shortest length is {0}, belongs to class {1}".format(shortest_time, class_name_0))
    print("The index of events stream (less than 2s) is {}".format(index_name))
    for i in range(19):
        # print("In class {0},\n the numbers of events stream shorter than "
        #       "2s is {1}\n".format(i, short_time_sum[i]))
        print(time_save[i])

    # for j in range(19):
    #     avg = sum(time_save[j])/len(time_save[j])
    #     print("class {0} The average time is {1} Min is {2}, Max is {3}".
    #           format(j, avg, min(time_save[j]), max(time_save[j])))


# #########build model############################################
def hyper_tuner(Aug):
    import time
    dataPath = "C:/Users/ASUS/OneDrive - Nanyang Technological University/datasets/DVSAnimals/data/"
    import tensorflow as tf
    start = time.time()
    "x_train.npy"
    if Aug:
        name_1 = "events_Aug/"
        name_2 = "Aug_"

    else:
        name_1 = "events/"
        name_2 = ""

    x_test = np.load(dataPath + "events/x_test.npy", allow_pickle=True)
    y_test = np.load(dataPath + "events/y_test.npy", allow_pickle=True)
    x_train = np.load(dataPath + name_1 + name_2 + "x_train.npy", allow_pickle=True)
    y_train = np.load(dataPath + name_1 + name_2 + "y_train.npy", allow_pickle=True)

    seed = 42
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = (
        train_test_split(x_train, y_train, test_size=0.125, random_state=seed))

    print('train shape{0}, validation shape {1},test shape {2}'.format
          (x_train.shape, x_val.shape, x_test.shape))

    # ########residual##########################
    from keras.callbacks import EarlyStopping
    from resnet_10 import ResNet18
    hypermodel = ResNet18(19)
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
    initial_learning_rate = 0.005
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        # initial_learning_rate,
        # # initial_learning_rate = 0.05,
        # decay_steps=5000,
        # decay_rate=0.5,
        initial_learning_rate=0.005,
        decay_steps=5000,
        decay_rate=0.8,

        staircase=True)
    # #########################
    hypermodel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        # optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"])
    # hypermodel.summary()
    # ############################# call back: the early stopping
    es = EarlyStopping(patience=30, restore_best_weights=True, monitor="val_accuracy")
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.1,
    #     patience=10,
    #     min_lr=1e-6,
    #     verbose=1)
    # ####check point path for model loading

    from event_stream import find_path
    root_dir, train_path, test_path = find_path()
    checkpoint_path = root_dir + "/Animals_3/cp.ckpt"
    import os
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    STEPS = 128

    bs = int(len(x_train) / STEPS)
    history = hypermodel.fit(x_train, y_train, batch_size=bs, steps_per_epoch=STEPS,
                             epochs=140,
                             # epochs=800,
                             validation_data=(x_val, y_val),
                             callbacks=[
                                 es,
                                 cp_callback
                             ]
                             )
    # # Create a callback that saves the model's weights###########
    hypermodel.summary()
    # plot loss during training
    from resnet_10 import plt_loss_acc
    plt_loss_acc(history)
    accuracy = hypermodel.evaluate(x_test, y_test, verbose=0)[1]
    print('Accuracy:', accuracy)

    # ####################################
    y_pred = np.array(list(map(lambda x: np.argmax(x), hypermodel.predict(x_test))))
    # ###############
    # hypermodel.load_weights("/training_1/cp.ckpt")
    # accuracy = hypermodel.evaluate(x_test, y_test, verbose=0)[1]
    # y_pred = hypermodel.predict(x_test)

    # ###########

    from sklearn.metrics import confusion_matrix, classification_report
    cm = confusion_matrix(y_test, y_pred)
    cm_normal = np.round(cm / np.sum(cm, axis=1).reshape(-1, 1), 2)

    classification_report(y_test, y_pred)
    fig_cm = plot_cm(cm_normal, accuracy, Aug)
    fig_cm.plot_cm()

    end = time.time()
    print('Training time: ', end - start)


class plot_cm:
    def __init__(self, cm, accuracy, Aug):
        self.cm = cm
        self.accuracy = accuracy
        if Aug:
            self.path = ('C:/Users/ASUS/OneDrive - Nanyang Technological University'
                         '/Figures/DVS_Animals/aug/')
        else:
            self.path = ('C:/Users/ASUS/OneDrive - Nanyang Technological University'
                         '/Figures/DVS_Animals/without aug/')

    def plot_cm(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(
            figsize=(25, 25)
        )
        TITLE_FONT_SIZE = {"size": "40"}
        LABEL_SIZE = 33
        sns.set(
            font_scale=2.4
        )

        g = sns.heatmap(self.cm, annot=True, fmt='g', cbar=False, cmap='Blues'
                        )
        g.set_xticks(range(19))
        g.set_yticks(range(19))
        # <--- set the ticks first

        axis_labels = ['cat', 'dog', 'camel', 'cow', 'sheep', 'goat', 'wolf',
                       'squirrel', 'mouse', 'dolphin', 'shark', 'lion', 'monkey',
                       'snake', 'spider', 'butterfly', 'bird', 'duck', 'zebra']

        # axis_labels = ["1","2","3","4", "5", "6", "7", "8", "9", "10", "11",
        #                "12", "13", "14", "15", "16", "17", "18", "19"]

        g.set_xticklabels(axis_labels)
        g.set_yticklabels(axis_labels)
        g.tick_params(axis="both", which="major",
                      labelsize=LABEL_SIZE
                      )
        plt.setp(g.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")
        plt.setp(g.get_yticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        plt.xlabel("Predicted", fontweight='bold',
                   fontdict=TITLE_FONT_SIZE
                   )
        plt.ylabel("Actual", fontweight='bold',
                   fontdict=TITLE_FONT_SIZE
                   )
        plt.title("Confusion Matrix", fontweight='bold',
                  fontdict=TITLE_FONT_SIZE
                  )
        g.text(40, -2.4, 'Accuracy {0}'.format(self.accuracy * 100), fontsize=14, color='black')
        plt.savefig(self.path + 'confusion_matrix' + str(self.accuracy * 100) + '.png')

        plt.show()
        plt.clf()


# sliced SL-Animals-DVS dataset definition
class AnimalsDvsSliced(Dataset):
    """
    The sliced Animals DVS dataset. Much faster loading and processing!
    Make sure to run "slice_data.py" for the 1st time before using this
    dataset to slice and save the files in the correct path.
    """

    def __init__(self, dataPath):
        from event_stream import c
        self.data_path = dataPath
        self.slicedDataPath = dataPath + 'sliced_recordings/'  # string
        self.files = list_sliced_files(np.loadtxt(dataPath + 'filelist.txt', dtype='str'))  # list [1121 files]
        # read class file
        self.classes = pd.read_csv(  # DataFrame
            dataPath + 'tags/SL-Animals-DVS_gestures_definitions.csv')
        self.index_array = index_arr()
        self.c = 3E-4
        # self.c= 5e-4 (acc: 78%)
        # self.c = 8E-4

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # load the sample file (NPY format), class name and index
        events, class_name, class_index, ss = self.get_sample(index)

        events = tonic.transforms.TimeAlignment()(events)
        # process the events
        """
        Use this method with Tonic frames. 
        """
        # print(class_name)
        return events, class_name, class_index

    def get_sample(self, index):
        # return the sample events, class name and class index of a sample
        assert index >= 0 and index <= 1120

        # the sample file name
        input_name = self.files[index]

        # load the sample file (NPY format)
        events = np.load(self.slicedDataPath + input_name)

        # find sample class
        class_index = index % 19  # [0-18]
        class_name = self.classes.iloc[class_index, 1]

        sensor_shape = (128, 128)

        return events, class_name, class_index, sensor_shape

    def Event_List(self, index):
        # events, label = self.Pack_Item()
        events, class_name, class_index = self.__getitem__(index)
        x0 = tuple(events["x"])
        y0 = tuple(events["y"])
        t = tuple(events["t"])
        p = tuple(events["p"])

        events_dict_pos = defaultdict(list)
        events_dict_neg = defaultdict(list)
        events_dict_polar = [events_dict_pos, events_dict_pos]
        events_dict_pos_time = defaultdict(list)
        events_dict_neg_time = defaultdict(list)
        events_dict_polar_time = [events_dict_pos_time, events_dict_neg_time]
        # j range(n): split event stream within ne parts,
        # Each part contains n event in whole event stream
        # rescale into same length:
        len_time = (t[-1] - t[0]) * 1e-6
        # print(len_time)

        # ##################original begin###########
        if len_time > 2:
            begin_n = int((len(t) - len(t) * 2 // int(len_time)) // 2)
            end_n = int(len(t) - begin_n)
            t = t[begin_n:end_n]
            x0 = x0[begin_n:end_n]
            y0 = y0[begin_n:end_n]
            p = p[begin_n:end_n]
        else:
            t = [x * (2 // len_time) for x in t]
        # ################original end###############

        # # ##############NEW VERSION BEGIN
        # if len_time >= 5:
        #     begin_n = int((len(t) - len(t) * 3 / int(len_time)) // 2)
        #     end_n = int(len(t) - begin_n)
        #     t = t[begin_n:end_n]
        #
        #     t = [x * 0.66 for x in t]
        #
        #     x0 = x0[begin_n:end_n]
        #     y0 = y0[begin_n:end_n]
        #     p = p[begin_n:end_n]
        #
        # elif 5 > len_time >= 2:
        #     begin_n = int((len(t) - len(t) * 2 / int(len_time)) // 2)
        #     end_n = int(len(t) - begin_n)
        #     t = t[begin_n:end_n]
        #     x0 = x0[begin_n:end_n]
        #     y0 = y0[begin_n:end_n]
        #     p = p[begin_n:end_n]
        #
        # else:
        #     t = [x * (2 // len_time) for x in t]
        #
        # # ##########NEW VERSION END

        end_t = (t[-1] - t[0]) * 1E-6

        for h in range(128 * 128):
            events_dict_pos[h] = []
            events_dict_neg[h] = []
            events_dict_pos_time[h] = []
            events_dict_neg_time[h] = []
            # print(len(t))
        for i in range(len(t)):
            key = self.index_array[x0[i]][y0[i]]
            if t[i] != t[-1] and i != 0:
                # if p[i]:
                #     events_dict_polar[0][key].append((t[i] - t[0]) * 1E-6 * self.c)
                # else:
                #     events_dict_polar[1][key].append((t[i] - t[0]) * 1E-6 * self.c)
                if p[i]:
                    events_dict_polar[0][key].append((t[i] - t[0]) * 1E-6)
                else:
                    events_dict_polar[1][key].append((t[i] - t[0]) * 1E-6)

        for m in range(2):
            events_dict = events_dict_polar[m]
            events_dict_time = events_dict_polar_time[m]
            for k in range(128 * 128):
                if events_dict[k]:
                    # for l in events_dict[k]:
                    for l in events_dict[k]:
                        if l == events_dict[k][0]:
                            l_0 = events_dict[k][0]
                        else:
                            events_dict_time[k].append((l - l_0) * self.c)
                            l_0 = l
                    # a = (end_t - events_dict[k][-1])*self.c
                    # if a>0.005:
                    # print(a)
                    events_dict_time[k].append((end_t - events_dict[k][-1]) * self.c)
        # return dict_pos, dict_neg, label, end_time
        return events_dict_pos_time, events_dict_neg_time, class_index

    # # unable to use
    # def Polarity_Match(self, index):
    #
    #     id_list, id_pulse_dict = generate_comparison_idpd_index()
    #     pos_temp_save = []
    #     neg_temp_save = []
    #     temp_save = [pos_temp_save, neg_temp_save]
    #     a1, a2, tau1, tau2, b1, b2, t1, t2, y1 = para_transistor_bi_exp()
    #     # for n in tqdm(range(data_num), desc="process"):
    #     output_arr = np.empty((128, 128, 2))
    #     dict_pos_time, dict_neg_time, label = self.Event_List(index)
    #     dict_list = [dict_pos_time, dict_neg_time]
    #     for polar in range(2):
    #         events_dict = dict_list[polar]
    #         # print("event list is {}".format(events_dict))
    #         for i in range(128 * 128):
    #             if events_dict[i]:
    #                 i_last = id_pulse_dict[1]
    #                 for j in events_dict[i]:
    #                     if j != events_dict[i][-1]:
    #                         # i_last = take_closest(id_list,
    #                         #                       id_time(i_last, j*self.c, a1, a2, tau1, tau2),
    #                         #                       id_pulse_dict)
    #                         i_last = take_closest(id_list,
    #                                               id_time(i_last, j, a1, a2, tau1, tau2),
    #                                               id_pulse_dict)
    #                     else:
    #                         # i_last = id_time(i_last, j*self.c, a1, a2, tau1, tau2)
    #                         i_last = id_time(i_last, j, a1, a2, tau1, tau2)
    #                 temp_save[polar].append(i_last)
    #             else:
    #                 temp_save[polar].append(0)
    #         for k in range(128):
    #             for m in range(128):
    #                 output_arr[k, m, polar] = temp_save[polar][int(self.index_array[m][k])]
    #     output_arr = np.rot90(output_arr, -1)
    #     np.save(self.data_path + "events/{0}.npy".format(index), output_arr)
    #     return label

    def Polarity_cal(self, index):
        import matplotlib.pyplot as plt
        i_d = calculate_match()
        pos_temp_save = []
        neg_temp_save = []
        temp_save = [pos_temp_save, neg_temp_save]
        # for n in tqdm(range(data_num), desc="process"):
        output_arr = np.empty((128, 128, 2))
        dict_pos_time, dict_neg_time, label = self.Event_List(index)
        dict_list = [dict_pos_time, dict_neg_time]
        for polar in range(2):
            events_dict = dict_list[polar]
            # print("event list is {}".format(events_dict))
            for i in range(128 * 128):
                if events_dict[i]:
                    id_last = i_d.d[0]
                    for j in events_dict[i]:
                        y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_, l_a, l_b = i_d.get_para(id_last)
                        # print(j*self.c)

                        id_b, id_a = id_time_new(id_last, j, y_0, A_1, A_2,
                                                 A_3, t_1, t_2, t_3, d_, l_a, l_b)

                        if j != events_dict[i][-1]:
                            id_last = id_a
                        else:
                            id_last = id_b
                    if id_last - 15.2 > 0:
                        temp_save[polar].append(id_last - 15.2)
                        # print(id_last)
                    else:
                        temp_save[polar].append(0)
                    # id_last = id_b - 15.2

                else:
                    temp_save[polar].append(0)
            for k in range(128):
                for m in range(128):
                    output_arr[k, m, polar] = temp_save[polar][int(self.index_array[m][k])]

        output_arr = np.rot90(output_arr, -1)

        # ################
        # plt.imshow(output_arr[:, :, 1])
        # plt.show()
        # ################
        np.save(self.data_path + "events/{0}.npy".format(index), output_arr)
        return label


def list_sliced_files(raw_file_list):
    # create a list of sliced files, given a list of 'raw' recording files
    sliced_file_list = []
    for file in raw_file_list:
        for i in range(19):
            sliced_file_list.append(file + '_{}.npy'.format(str(i).zfill(2)))

    return sliced_file_list


class gen_train_test:

    def __init__(self, data_num, dataPath):
        self.data_num = data_num
        self.dataPath = dataPath
        self.index_labels = np.load(self.dataPath + "events/dataset_labels.npy", allow_pickle=True)

    def stack_frames(self):
        from sklearn.model_selection import train_test_split

        index_class_list = self.search()
        for i in tqdm(range(19), desc="Stack_class"):
            stack_class = np.zeros((59, 128, 128, 2))
            # print("class name is: {}".format(i))
            for j in range(len(index_class_list[i])):
                # print("number {0}, belongs to class {1}".format(index_class_list[i][j], i))
                frame = np.load(self.dataPath + "events/{}.npy".format(index_class_list[i][j]), allow_pickle=True)
                stack_class[j, :, :, :] = frame
            stack_labels = [i] * 59
            X_train, X_test, y_train, y_test = train_test_split(stack_class, stack_labels,
                                                                test_size=0.3, random_state=42, shuffle=True)
            # print("the shape: x_train {}\n y_train {}\n, x_test {}\n, y_test {}".
            #       format(X_train.shape, len(y_train), X_test.shape, len(y_test)))
            if i == 0:
                ouput_x_train = np.zeros((19 * len(y_train), 128, 128, 2))
                ouput_y_train = np.zeros((19 * len(y_train),))
                ouput_x_test = np.zeros((19 * len(y_test), 128, 128, 2))
                ouput_y_test = np.zeros((19 * len(y_test),))

            ouput_x_train[i * len(y_train): (i + 1) * len(y_train), :, :, :] = X_train
            ouput_y_train[i * len(y_train): (i + 1) * len(y_train), ] = y_train
            ouput_x_test[i * len(y_test): (i + 1) * len(y_test), :, :, :] = X_test
            ouput_y_test[i * len(y_test): (i + 1) * len(y_test), ] = y_test

        np.save(self.dataPath + "events/x_train.npy", ouput_x_train)
        np.save(self.dataPath + "events/y_train.npy", ouput_y_train)
        np.save(self.dataPath + "events/x_test.npy", ouput_x_test)
        np.save(self.dataPath + "events/y_test.npy", ouput_y_test)

        print("End: Stack by class")

    def stack_aug(self):
        from frames_processing import aug_process
        x_train = np.load(self.dataPath + "events/x_train.npy", allow_pickle=True)
        y_train = np.load(self.dataPath + "events/y_train.npy", allow_pickle=True)
        print(y_train.shape)
        print(x_train.shape)
        aug_y_train = np.zeros((len(y_train) * 6,))
        aug_x_train = np.zeros((len(y_train) * 6, 128, 128, 2))
        for i in tqdm(range(len(y_train)), desc="Augmentation"):
            aug_x, aug_y = aug_process(x_train=x_train[i], y_train=y_train[i])
            np.save(self.dataPath + "events_Aug/Aug_{}.npy".format(i), aug_x)
            aug_y_train[i * 6:i * 6 + 6, ] = aug_y
        np.save(self.dataPath + "events_Aug/Aug_y_train.npy", aug_y_train)

        for i in tqdm(range(len(y_train)), desc="Stack Augmentation"):
            aug_x_train[i * 6:i * 6 + 6, :, :, :] = np.load(self.dataPath + "events_Aug/Aug_{}.npy".
                                                            format(i), allow_pickle=True)
        np.save(self.dataPath + "events_Aug/Aug_x_train.npy", aug_x_train)
        print("End: Augmentation")

    def search(self):
        index_class_list = defaultdict(list)
        # index_class_num = defaultdict(list)

        for j in tqdm(range(19), desc="Search"):
            # a=0
            for i in range(self.data_num):
                if j == self.index_labels[i]:
                    # print(class_name)
                    # print(self.index_labels[i])
                    # a += 1
                    index_class_list[j].append(i)
            # index_class_num[j] = a
        # print("Class 10 contains {}".format(index_class_list[10]))
        # print("the index number of corresponding labels are :{}".format(index_class_num))
        print("End: Search")
        return index_class_list


def slice(data_path):
    # A text file with a list of the 'raw' file names
    file_list = data_path + 'filelist.txt'

    # create sliced dataset directory and path
    os.makedirs(data_path + "sliced_recordings", exist_ok=True)
    sliced_data_path = data_path + "sliced_recordings/"

    # load file names into a 1D array
    files = np.loadtxt(file_list, dtype='str')  # 1D array [max 59]

    # check if dataset is already sliced
    if len(os.listdir(sliced_data_path)) < (19 * len(files)):

        print('Slicing the dataset, this may take a while...')

        # For each of the raw recordings: slice in 19 pieces and save to disk
        for record_name in files:
            print('Processing record {}...'.format(record_name))

            # read the DVS file
            """
            The output of this function:
                sensor_shape: tuple, the DVS resolution (128, 128)
                events: 1D-array, the sequential events on the file
                        1 microsecond resolution
                        each event is 4D and has the shape 'xytp'
            """
            # sensor_shape, events = tonic.io.read_dvs_128(data_path + 'recordings/'
            #                                              + record_name + '.aedat')
            sensor_shape, events = read_dvs_128(data_path + 'recordings/' + record_name + '.aedat')
            # read the tag file
            tagfile = pd.read_csv(data_path + 'tags/' + record_name + '.csv')  # df

            # define event boundaries for each class
            events_start = list(tagfile["startTime_ev"])
            events_end = list(tagfile["endTime_ev"])

            # create a list of arrays, separating the recording in 19 slices
            sliced_events = tonic.slicers.slice_events_at_indices(events,
                                                                  events_start,
                                                                  events_end)
            # save 19 separate events on disk
            for i, chosen_slice in enumerate(sliced_events):
                np.save(sliced_data_path + '{}_{}.npy'.format(
                    record_name, str(i).zfill(2)), chosen_slice)
        print('Slicing completed.\n')

    else:
        print('Dataset is already sliced.\n')


def read_dvs_128(filename):
    """Get the aer events from DVS with resolution of rows and cols are (128, 128)

    Parameters:
        filename: filename

    Returns:
        shape (tuple):
            (height, width) of the sensor array
        xytp: numpy structured array of events
    """
    print(filename)
    data_version, data_start, start_timestamp = read_aedat_header_from_file(filename)

    all_events = tonic.io.get_aer_events_from_file(filename, data_version, data_start)
    all_addr = all_events["address"]
    t = all_events["timeStamp"]

    x = (all_addr >> 8) & 0x007F
    y = (all_addr >> 1) & 0x007F
    p = all_addr & 0x1

    xytp = tonic.io.make_structured_array(x, y, t, p)
    shape = (128, 128)
    return shape, xytp


def read_aedat_header_from_file(filename):
    """Get the aedat file version and start index of the binary data.

    Parameters:
        filename (str):     The name of the .aedat file

    Returns:
        data_version (float):   The version of the .aedat file
        data_start (int):       The start index of the data
        start_timestamp (int):  The start absolute system timestamp in micro-seconds
    """
    filename = os.path.expanduser(filename)
    print(filename)

    assert os.path.isfile(filename), f"The .aedat file '{filename}' does not exist."
    f = open(filename, "rb")
    count = 1
    is_comment = "#" in str(f.read(count))

    start_timestamp = None
    while is_comment:
        # Read the rest of the line
        head = str(f.readline())
        if "!AER-DAT" in head:
            data_version = float(head[head.find("!AER-DAT") + 8: -5])
        elif "Creation time:" in head:
            start_timestamp = int(head.split()[4].split("\\")[0])
        is_comment = "#" in str(f.read(1))
        count += 1
    data_start = f.seek(-1, 1)
    f.close()
    return data_version, data_start, start_timestamp
