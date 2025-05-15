from params_adjustment import polar_remove_load
from params_adjustment import plot_cm, plt_loss_acc
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, GlobalAveragePooling2D, BatchNormalization, Layer, Add, \
    Dropout,LSTM,ReLU
from keras.models import Model
from keras import Input
from resnet_10 import ResnetBlock
import tensorflow as tf
import numpy as np


def hyper_tuner(aug,name,dir_name, suffix, mode=False,resnet_num=3,model_path="results"):
    import os
    os.makedirs(dir_name, exist_ok=True)
    import time
    start = time.time()
    random_state = 86
    train_validation_rate = 0.125

    x_train, x_test, y_train, y_test, x_val, y_val = polar_remove_load(aug, random_state, train_validation_rate, suffix,
                                                                       mode)

    bs = 100
    print("batch size: ", bs)


    print('train shape{0}, validation shape {1},test shape {2}'.format
          (x_train.shape, x_val.shape, x_test.shape))
    # ########residual##########################
    from keras.callbacks import EarlyStopping
    from six_channel_event_frame import n_num
    hypermodel = ResNetLSTM(10,n_num)
    # from tensorflow.keras.applications import ResNet50
    # hypermodel=ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(128, 128, 2)))
    # hypermodel=None_ResnetBlock(2);
    # #print the model# ############
    hypermodel.build(input_shape=(None,n_num, 128, 128, 2))
    hypermodel.build_graph().summary()
    # tf.keras.utils.plot_model(
    #     hypermodel.build_graph(),  # here is the trick (for now)
    #     to_file='model.png', dpi=96,  # saving
    #     show_shapes=True, show_layer_names=True,  # show shapes and layer name
    #     expand_nested=False  # will show nested block
    # )
    # ################## learning rate scheduler
    import random
    initial_learning_rate = 0.001
    decay_steps = 5000
    decay_rate = 0.5
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

    STEPS = 128
    bs = int(len(x_train) / STEPS)


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
        , "random_seed": random_state, "test_size": train_validation_rate}
    with open(f"{dir_name}/{name}.txt", "w") as file:
        file.write(",\n".join([f"{name}: {value}" for name, value in savepara.items()]))
    hypermodel.save_weights(f"{dir_name}/{name}.h5")
    end = time.time()
    print('Training time: ', end - start)

    return x_train, x_val, x_test, y_train, y_val, y_test

class ResNetLSTM(Model):

    def __init__(self, num_classes,time_steps=3,resnet_num=3,**kwargs):
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
        self.resnet_num=resnet_num
        self.time_steps=time_steps
        self.avg_pool = GlobalAveragePooling2D()
        self.merge = Add()
        self.flat = Flatten()
        self.fc = Dense(num_classes, activation="softmax")
        self.avg_pool = GlobalAveragePooling2D()

        # LSTM部分
        self.lstm = LSTM(128, return_sequences=False)  # 128 维隐藏层
        self.fc = Dense(num_classes, activation="softmax")
        # self.fc = Dense(10, activation="softmax")
    def call(self, inputs):
        feature_list = []
        for t in range(self.time_steps):
            out = self.conv_1(inputs[:,t,:,:,:])
            out = self.init_bn(out)
            out = ReLU()(out)
            out_0 = self.pool_2(out)
            out = self.res_1_1(out_0)
            out = self.res_1_2(out)
            out = self.merge([out, out_0])
            out = self.res_2_1(out)

            # 3 more resnet
            # out = self.res_2_2(out)
            # out = self.res_3_1(out)
            # out = self.res_3_2(out)

            out = self.avg_pool(out)
            feature_list.append(out)
        features = tf.stack(feature_list, axis=1)
        out = self.lstm(features)
        out = self.fc(out)
        return out


    def build_graph(self):
        x = Input(shape=(self.time_steps,128, 128, 2))
        return Model(inputs=[x], outputs=self.call(x))