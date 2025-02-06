import os

import numpy as np
import tensorflow as tf
from spikingjelly.activation_based.model.tv_ref_classify.utils import accuracy


class Model:
    def __init__(self,model_path):
        from resnet_10 import ResNet18
        self.model_path = model_path
        self.x_test,self.y_test=self.load_test_set()
        self.x_train,self.y_train=self.load_train_set()
        self.label_for_each_class_train= [[i for i in range(len(self.y_train)) if self.y_train[i] == j] for j in
                                          set(self.y_train)]
        self.label_len=len(self.y_test)
        self.label_for_each_class_test=[[i for i in range(len(self.y_test)) if self.y_test[i] == j] for j in set(self.y_test)]
        hypermodel=ResNet18(10)
        hypermodel.build(input_shape=(None, 128, 128, 2))
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            # initial_learning_rate,
            initial_learning_rate=0.01,
            decay_steps=3000,
            decay_rate=0.5,
            staircase=True)
        hypermodel.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
            # optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
            loss='sparse_categorical_crossentropy',
            metrics=["accuracy"])
        hypermodel.load_weights(self.model_path)
        self.predict_y_test_res = np.argmax(hypermodel.predict(self.x_test),axis=1)
        self.predict_y_test_vote =  hypermodel.predict(self.x_test)
        self.accuracy=hypermodel.evaluate(self.x_test, self.y_test, verbose=0)[1]
        print()



    def load_test_set(self):
        from event_stream import find_path
        import numpy as np
        root_dir, train_path, test_path = find_path()
        if "tune" in self.model_path:
            x_test=np.load(test_path+ "tuned_dataset_features.npy", allow_pickle=True)
            y_test=np.load(test_path+ "tuned_dataset_labels.npy", allow_pickle=True)

            from improvement_tuning import cal_set_7
            # b_6, e_6, b_7, e_7 = cal_set_7(y_test)
            # x_test = np.concatenate([x_test[:e_7 - e_6 + b_6, :, :, :], x_test[e_7 + 1:, :, :, :]], axis=0)
            # y_test = np.concatenate([y_test[:e_7 - e_6 + b_6, ], y_test[e_7 + 1:, ]], axis=0)
            from frames_processing import frame_normalization
            x_train, x_test = frame_normalization(x_test, x_test)
        else:
            x_test = np.load(os.path.join(test_path, "dataset_features.npy"), allow_pickle=True)
            y_test =np.load(os.path.join(test_path, "dataset_labels.npy"), allow_pickle=True)
        return x_test,y_test

    def get_idx_of_misclassified_labels(self):
        return [i for i in range(self.label_len) if self.predict_y_test_res[i] != self.y_test[i]]

    def get_idx_of_misclassified_labels_by_class(self,n:int):
        return set(self.get_idx_of_misclassified_labels()) & set(self.label_for_each_class_test[n])

    def show_img(self,img_idx,title:str=None,mode:str="test"):
        import event_stream
        import matplotlib.pyplot as plt
        tags = ['Positive', 'Negative']
        cmap = 'viridis'
        if mode=="test":
            img=self.x_test[img_idx]
        elif mode == "train":
            img=self.x_train[img_idx]
        output_arr = np.array([img[:,:,0], img[:,:,1]])

        fig, ax = plt.subplots(1, 2)
        plt.suptitle("{} {}".format(title,img_idx))
        ax[0].imshow(output_arr[0, :, :], cmap=cmap)
        ax[0].set_title(tags[0])
        ax[1].imshow(output_arr[1, :, :], cmap=cmap)
        ax[1].set_title(tags[1])
        fig.tight_layout()
        # fig.colorbar
        # print(output_arr[:, :, 0]*1E5)
        fig.show()

    def load_train_set(self):
        from event_stream import find_path
        import numpy as np
        root_dir, train_path, test_path = find_path()
        if "tune" not in self.model_path:
            x_train=np.load(os.path.join(train_path, "Aug_dataset_features.npy"), allow_pickle=True)
            y_train=np.load(os.path.join(train_path, "Aug_dataset_labels.npy"), allow_pickle=True)
            return x_train,y_train
        x_train=np.load(os.path.join(train_path, "tuned_Aug_dataset_features.npy"), allow_pickle=True)
        y_train=np.load(os.path.join(train_path, "tuned_Aug_dataset_labels.npy"), allow_pickle=True)
        return x_train,y_train

