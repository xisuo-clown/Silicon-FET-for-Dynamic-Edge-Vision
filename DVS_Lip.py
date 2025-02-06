import numpy as np
from tqdm import tqdm

from event_stream import generate_comparison_idpd_index, para_transistor_bi_exp, take_closest, id_time, index_arr
import numpy as np
import IPython
from keras import Input

from keras.callbacks import EarlyStopping
from keras.layers import (Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D,
                          BatchNormalization, Add)
from keras.models import Model
import tensorflow as tf


def initial_load(train: bool = True):
    import tonic
    path = path_list(train)
    dataset = tonic.datasets.DVSLip(path.root_dir, train)
    # # 100 labels
    if train:
        a = "Train"
    else:
        a = "Test"
    data_num = dataset.__len__()
    print("The length of {0} is {1} ".format(a, data_num))
    # path for event array saving: train and test
    # the length of data: 14876 (Train)/ 4975 (Test)
    return dataset, data_num


def gen_aug():
        import frames_processing
        frames_processing.gen_augmentation_frame_new()

def gen_aug_stack_frame():
        import frames_processing
        frames_processing.gen_stack_frame_new(True)




def Collect_Frames():
    index_array = index_arr()
    c = 1.3E-3
    train_list = [True, False]

    for i in train_list:
        dataset, data_num = initial_load(i)
        path = path_list(i)
        labels_list = []
        for j in tqdm(range(data_num), desc="Frames Generation"):
            frame_n = DVS_Lip(dataset, j, index_array, c, i, path.present_path())
            labels_list.append(frame_n.Polarity_Match())
        print("The length of the label list is {0}, and the corresponding data"
              " number is {1}".format(len(labels_list), data_num))
        np.save(path.present_path() + "dataset_labels.npy", labels_list)
        stack_array = np.empty((data_num, 128, 128, 2))
        for j in tqdm(range(data_num), desc="Frames Stacking"):
            stack_array[j, :, :, :] = np.load(path.present_path() + "{0}.npy".format(j), allow_pickle=True)
        np.save(path.present_path() + "dataset_features.npy", stack_array)


# #########build model############################################
def hyper_tuner():
    import time
    import os
    import tensorflow as tf
    start = time.time()

    x_test = np.load(os.path.join(path_list(False).present_path(), "dataset_features.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(path_list(False).present_path(), "dataset_labels.npy"), allow_pickle=True)
    x_train = np.load(os.path.join(path_list(True).present_path(), "dataset_features.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(path_list(True).present_path(), "dataset_labels.npy"), allow_pickle=True)
    seed = 42
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = (
        train_test_split(x_train, y_train, test_size=0.125, random_state=seed))

    print('train shape{0}, validation shape {1},test shape {2}'.format
          (x_train.shape, x_val.shape, x_test.shape))

    # ########residual##########################
    from keras.callbacks import EarlyStopping
    hypermodel = ResNet18(100)
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
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        # initial_learning_rate = 0.05
        decay_steps=3000,
        decay_rate=0.5,
        staircase=True)
    # #########################
    hypermodel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        # optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"])
    # hypermodel.summary()
    # ############################# call back: the early stopping
    es = EarlyStopping(patience=45, restore_best_weights=True, monitor="val_accuracy")
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.1,
    #     patience=10,
    #     min_lr=1e-6,
    #     verbose=1)
    # ####check point path for model loading

    from event_stream import find_path
    root_dir, train_path, test_path = find_path()
    checkpoint_path = root_dir + "/lip_0/cp.ckpt"
    import os
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    STEPS = 96
    bs = int(len(x_train) / STEPS)
    history = hypermodel.fit(x_train, y_train, batch_size=bs, steps_per_epoch=STEPS,
                             epochs=140,
                             # epochs=800,
                             validation_data=(x_val, y_val),
                             callbacks=[
                                 # es,
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
    fig_cm = plot_cm(cm_normal, accuracy)
    fig_cm.plot_cm()

    end = time.time()
    print('Training time: ', end - start)

def hyper_tuner_aug():
    import time
    import os
    import tensorflow as tf
    start = time.time()

    x_test = np.load(os.path.join(path_list(False).present_path(), "dataset_features.npy"), allow_pickle=True)
    y_test = np.load(os.path.join(path_list(False).present_path(), "dataset_labels.npy"), allow_pickle=True)
    x_train = np.load(os.path.join(path_list(True).present_path(), "trainAug_dataset_features.npy"), allow_pickle=True)
    y_train = np.load(os.path.join(path_list(True).present_path(), "trainAug_dataset_labels.npy"), allow_pickle=True)
    seed = 42
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = (
        train_test_split(x_train, y_train, test_size=0.125, random_state=seed))

    print('train shape{0}, validation shape {1},test shape {2}'.format
          (x_train.shape, x_val.shape, x_test.shape))

    # ########residual##########################
    from keras.callbacks import EarlyStopping
    hypermodel = ResNet18(100)
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
    initial_learning_rate = 0.01
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        # initial_learning_rate = 0.05
        decay_steps=3000,
        decay_rate=0.5,
        staircase=True)
    # #########################
    hypermodel.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
        # optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"])
    # hypermodel.summary()
    # ############################# call back: the early stopping
    es = EarlyStopping(patience=45, restore_best_weights=True, monitor="val_accuracy")
    # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    #     monitor='val_loss',
    #     factor=0.1,
    #     patience=10,
    #     min_lr=1e-6,
    #     verbose=1)
    # ####check point path for model loading

    from event_stream import find_path
    root_dir, train_path, test_path = find_path()
    checkpoint_path = root_dir + "/lip_0/cp.ckpt"
    import os
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)

    STEPS = 96
    bs = int(len(x_train) / STEPS)
    history = hypermodel.fit(x_train, y_train, batch_size=bs, steps_per_epoch=STEPS,
                             epochs=140,
                             # epochs=800,
                             validation_data=(x_val, y_val),
                             callbacks=[
                                 # es,
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
    fig_cm = plot_cm(cm_normal, accuracy)
    fig_cm.plot_cm()

    end = time.time()
    print('Training time: ', end - start)
class DVS_Lip:
    # file_path = os.path.join(path, self.folder_name)
    def __init__(self, dataset, n, index_array, c, train, save_path):
        self.dataset = dataset
        self.n = n
        self.index_array = index_array
        # the parameter
        self.c = c
        self.train = train
        self.save_path = save_path

    def Pack_Item(self):
        events, target = self.dataset[self.n]
        # print("The corresponding class is {}".format(target))
        return events, target


    def Event_List(self):
        events, label = self.Pack_Item()
        x0 = tuple(events["x"])
        y0 = tuple(events["y"])
        t = tuple(events["t"])
        p = tuple(events["p"])
        from collections import defaultdict
        events_dict_pos = defaultdict(list)
        events_dict_neg = defaultdict(list)
        events_dict_polar = [events_dict_pos, events_dict_pos]

        events_dict_pos_time = defaultdict(list)
        events_dict_neg_time = defaultdict(list)
        events_dict_polar_time = [events_dict_pos_time, events_dict_neg_time]
        # j range(n): split event stream within ne parts,
        # Each part contains n event in whole event stream
        end_t = (t[-1] - t[0]) * 1E-6 * self.c
        for h in range(128 * 128):
            events_dict_pos[h] = []
            events_dict_neg[h] = []
            events_dict_pos_time[h] = []
            events_dict_neg_time[h] = []
            # print(len(t))
        for i in range(len(t)):
            key = self.index_array[x0[i]][y0[i]]
            if t[i] != t[-1] and i != 0:
                if p[i]:
                    events_dict_polar[0][key].append((t[i] - t[0]) * 1E-6 * self.c)
                else:
                    events_dict_polar[1][key].append((t[i] - t[0]) * 1E-6 * self.c)

        for m in range(2):
            events_dict = events_dict_polar[m]
            events_dict_time = events_dict_polar_time[m]
            for k in range(128 * 128):
                if events_dict[k]:
                    # for l in events_dict[k]:
                    for l in events_dict[k]:
                        if l != events_dict[k][0]:
                            events_dict_time[k].append(l - l_0)
                            l_0 = l
                        else:
                            l_0 = events_dict[k][0]
                        events_dict_time[k].append(end_t - events_dict[k][-1])
        # return dict_pos, dict_neg, label, end_time
        return events_dict_pos_time, events_dict_neg_time, label

    def test(self):
        events, label = self.Pack_Item()
        x0 = tuple(events["x"])
        y0 = tuple(events["y"])
        t = tuple(events["t"])
        p = tuple(events["p"])
        from collections import defaultdict
        events_dict_pos = defaultdict(list)
        events_dict_neg = defaultdict(list)
        events_dict_polar = [events_dict_pos, events_dict_pos]

        events_dict_pos_time = defaultdict(list)
        events_dict_neg_time = defaultdict(list)
        events_dict_polar_time = [events_dict_pos_time, events_dict_neg_time]
        # j range(n): split event stream within ne parts,
        # Each part contains n event in whole event stream
        end_t = (t[-1] - t[0]) * 1E-6 * self.c
        print("No.{0} end t is {1}, t[0] is {2}".format(self.n, end_t, t[0]))

    # Save the positive & negative, generate 30 samples from per video
    def Polarity_Match(self):
        id_list, id_pulse_dict = generate_comparison_idpd_index()
        pos_temp_save = []
        neg_temp_save = []
        temp_save = [pos_temp_save, neg_temp_save]
        a1, a2, tau1, tau2, b1, b2, t1, t2, y1 = para_transistor_bi_exp()
        # for n in tqdm(range(data_num), desc="process"):
        output_arr = np.empty((128, 128, 2))
        dict_pos_time, dict_neg_time, label = self.Event_List()
        dict_list = [dict_pos_time, dict_neg_time]
        for polar in range(2):
            events_dict = dict_list[polar]
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
                    output_arr[k, m, polar] = temp_save[polar][int(self.index_array[m][k])]
            np.save(self.save_path + "{0}.npy".format(self.n), output_arr)
        return label


class path_list:
    def __init__(self, train):
        self.save_path = None
        self.train = train
        import os
        self.root_dir =  os.getcwd()

    def present_path(self):
        import os
        root_dir = os.getcwd()
        if self.train:
            save_path = (root_dir + '/DvsLip/events_frames/train/')
        else:
            save_path = (root_dir + '/DvsLip/events_frames/test/')
        return save_path


class plot_cm:
    def __init__(self, cm, accuracy):
        self.cm = cm
        self.accuracy = accuracy

    def plot_cm(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(
            figsize=(50, 50)
        )
        TITLE_FONT_SIZE = {"size": "22"}
        LABEL_SIZE = 13
        sns.set(
            font_scale=0.6
        )

        g = sns.heatmap(self.cm, annot=True, fmt='g', cbar=False, cmap='Blues'
                        )
        g.set_xticks(range(100))
        g.set_yticks(range(100))
        # <--- set the ticks first
        # g.set_xticklabels(axis_labels)
        # g.set_yticklabels(axis_labels)
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
        plt.savefig('plot1.png')
        plt.show()
        plt.clf()


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


class None_ResnetBlock(Model):
    """
    A standard resnet block.
    """

    def __init__(self, channels: int):
        """
        channels: same as number of convolution kernels
        """
        super().__init__()

        self.__channels = channels
        self.__strides = [2, 1]

        KERNEL_SIZE = (3, 3)
        # use He initialization, instead of Xavier (a.k.a 'glorot_uniform' in Keras), as suggested in [2]
        INIT_SCHEME = "he_normal"

        self.conv_1 = Conv2D(self.__channels, strides=self.__strides[0],
                             kernel_size=KERNEL_SIZE, padding="same", kernel_initializer=INIT_SCHEME)
        self.bn_1 = BatchNormalization()

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.bn_1(x)
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
        self.conv_1 = Conv2D(filters[0],
                             (4, 4),
                             strides=2,
                             padding="same", kernel_initializer="he_normal")
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
        out = tf.nn.relu(out)
        out_0 = self.pool_2(out)
        out = self.res_1_1(out_0)
        out = self.res_1_2(out)
        out_1 = self.merge([out, out_0])
        out = self.res_2_1(out_1)
        out = self.res_2_2(out)
        out = self.res_3_1(out)
        out = self.res_3_2(out)
        #####with extra shortcut end
        out = self.avg_pool(out)
        out = self.flat(out)
        out = self.fc(out)
        return out

    def build_graph(self):
        x = Input(shape=(128, 128, 2))
        return Model(inputs=[x], outputs=self.call(x))


# callback to clear the training output
class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)
