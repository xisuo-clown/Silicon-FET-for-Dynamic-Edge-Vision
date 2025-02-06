# optional optimization process: remove class 7 and data augmentation


# data augmentation: optional
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


def gen_augmentation_frame_new():
    label_path="dataset_labels.npy"
    prefix=""
    import os
    import numpy as np
    import time
    from event_stream import find_path
    begin = time.time()
    root_dir, train_path, test_path = (os.getcwd()+r"/DvsLip", os.getcwd()+r"/DvsLip/events_frames/train",
                                       os.getcwd()+r"/DvsLip/events_frames/test")
    from tqdm import tqdm
    from sklearn.preprocessing import LabelEncoder
    # Augmentation: the training set (aug.npy and label index)
    y_labelencoder = LabelEncoder()
    y = np.load(os.path.join(train_path, label_path), allow_pickle=True)

    y_train = y_labelencoder.fit_transform(y)
    y_train = y_train.tolist()

    for i in tqdm(range(len(y_train)), desc="Rotate and shift"):
        path = os.path.join(train_path, f'{prefix}{i}.npy')
        x_train = np.load(path, allow_pickle=True)
        x, y = aug_process(x_train, y_train[i])
        # os.remove(path)
        if i == 0:
            y_list = y
        else:
            y_list = np.concatenate((y_list[:, ], y[:, ]), axis=0)
        np.save(train_path + f"Aug_{prefix}{i}.npy", x)
    print('y_train shape: should be {0}, len_y is {1}'.format(6 * len(y_train), y_list.shape))
    np.save(train_path + "Aug_dataset_labels.npy", y_list)
    # stack the frames: train and test
    gen_stack_frame(Aug=True)
    print('END: Stacking')
    end = time.time()
    print(end - begin)


# augmentation process
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

def gen_stack_frame_new(Aug: bool,label_path="", prefix=""):
    import os
    import numpy as np
    import time
    begin = time.time()
    root_dir, train_path, test_path = (os.getcwd()+r"/DvsLip", os.getcwd()+r"/DvsLip/events_frames/train",
                                       os.getcwd()+r"/DvsLip/events_frames/test")
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
            path = path_type[j]+f'{name_load[j]}{i}.npy'
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
# stack frames
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


# remove class 7 (amount of training and test samples is twice larger than other classes)
def polar_remove_set7(aug):
    import os
    from event_stream import find_path
    root_dir, train_path, test_path = find_path()
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


def polar_remove_set7_load(aug):
    import os
    from event_stream import find_path
    root_dir, train_path, test_path = find_path()
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
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    from sklearn.model_selection import train_test_split
    # max_value = np.max(x_train)
    # min_value = np.min(x_train)
    # print('train set max {0}, min {1}'.format(max_value, min_value))
    # max_value = np.max(x_test)
    # min_value = np.min(x_test)
    # print('test set max {0}, min {1}'.format(max_value, min_value))
    #
    # x_train = x_train * 255
    # x_test = x_test * 255
    # print(np.max(x_test), np.max(x_train))
    #
    # print(x_test.shape, x_train.shape)
    # # print('mean value: test {0}, train{1}'.format(np.mean(x_test[:360,:,:,:]),
    # #                                               np.mean(x_train[:1700,:,:,:])))

    x_train, x_test = frame_normalization(x_train, x_test)
    seed = 42
    x_train, x_val, y_train, y_val = (
        train_test_split(x_train, y_train, test_size=0.125, random_state=seed))
    return x_train, x_test, y_train, y_test, x_val, y_val


# loading data
def polar_remove_load(aug,random_state=86,train_validation_rate=0.125):
    import os
    from event_stream import find_path
    root_dir, train_path, test_path = find_path()
    # processing
    import numpy as np
    if aug:
        name = 'Aug_'
    else:
        name = 'Ori_'

    x_test = np.load(os.path.join(test_path, "dataset_features.npy".format(name)), allow_pickle=True)
    y_test = np.load(os.path.join(test_path, "dataset_labels.npy".format(name)), allow_pickle=True)
    x_train = np.load(os.path.join(train_path, "{0}dataset_features.npy".format(name)), allow_pickle=True)
    y_train = np.load(os.path.join(train_path, "{0}dataset_labels.npy".format(name)), allow_pickle=True)
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


# def polar_resplit_load(aug):
#     import os
#     from event_stream import find_path
#     root_dir, train_path, test_path = find_path()
#     # processing
#     import numpy as np
#     if aug:
#         name = 'Aug_'
#     else:
#         name = 'Ori_'
#     x_test = np.load(os.path.join(test_path, "dataset_features_remove.npy".format(name)), allow_pickle=True)
#     y_test = np.load(os.path.join(test_path, "dataset_labels_remove.npy".format(name)), allow_pickle=True)
#     x_train = np.load(os.path.join(train_path, "{0}dataset_features_remove.npy".format(name)), allow_pickle=True)
#     y_train = np.load(os.path.join(train_path, "{0}dataset_labels_remove.npy".format(name)), allow_pickle=True)
#
#     print(x_train.shape)
#     print(y_train)
#
#     for i in range(10):
#         con = 0
#         for j in range(len(y_train - 1)):
#             if y_train[j] == i:
#                 con += 1
#             if con == 1:
#                 j_0 = j
#                 print('j0 {}'.format(j_0))
#             if i != 9 and y_train[j] == i and y_train[j + 1] != i:
#                 j_tra = j + 1
#             elif i == 9:
#                 j_tra = len(y_train - 1)
#
#         if i == 0:
#             x_te = x_train[j_0:j_0 + 60, :, :, :]
#             y_te = y_train[j_0:j_0 + 60, ]
#             x_val = x_train[j_0 + 60:j_0 + 90, :, :, :]
#             y_val = y_train[j_0 + 60:j_0 + 90, ]
#             x_tra = x_train[j_0 + 90:j_tra, :, :, :]
#             y_tra = y_train[j_0 + 90:j_tra, ]
#         else:
#             x_te = np.concatenate([x_te[:, :, :, :], x_train[j_0:j_0 + 60, :, :, :]], axis=0)
#             y_te = np.concatenate((y_te[:], y_train[j_0:j_0 + 60, ]), axis=0)
#             x_val = np.concatenate([x_val[:, :, :, :], x_train[j_0 + 60:j_0 + 90, :, :, :]], axis=0)
#             y_val = np.concatenate((y_val[:], y_train[j_0 + 60:j_0 + 90, ]), axis=0)
#             x_tra = np.concatenate([x_tra[:, :, :, :], x_train[j_0 + 90:j_tra, :, :, :]], axis=0)
#             y_tra = np.concatenate((y_tra[:], y_train[j_0 + 90:j_tra, ]), axis=0)
#
#         print('{0} test shape {1}, test list {2}'.format(i, x_te.shape, y_te.shape))
#         print('{0} val shape {1}, val list {2}'.format(i, x_val.shape, y_val.shape))
#         print('{0} train shape {1}, train list {2}'.format(i, x_tra.shape, y_tra.shape))
#
#     # print('{0}: class {1} have {2} data'.format(dict_name[d], i, h))
#     # print('{0}: feature shape {1}'.format(dict_name[d], x.shape))
#
#     from sklearn.utils import shuffle
#     seed = 42
#     x_train, y_train = shuffle(x_tra, y_tra, random_state=seed)
#     x_val, y_val = shuffle(x_val, y_val, random_state=seed)
#     x_test, y_test = shuffle(x_te, y_te, random_state=seed)
#     return x_train, x_test, y_train, y_test, x_val, y_val


def frame_normalization(x_train, x_test):
    import numpy as np
    from sklearn.model_selection import train_test_split

    max_value = np.max(x_train)
    min_value = np.min(x_train)
    print('train set max {0}, min {1}'.format(max_value, min_value))
    max_value = np.max(x_test)
    min_value = np.min(x_test)
    print('test set max {0}, min {1}'.format(max_value, min_value))

    x_train = x_train * 255
    x_test = x_test * 255
    print(np.max(x_test), np.max(x_train))

    print(x_test.shape, x_train.shape)
    # print('mean value: test {0}, train{1}'.format(np.mean(x_test[:360,:,:,:]),
    #                                               np.mean(x_train[:1700,:,:,:])))
    return x_train, x_test
