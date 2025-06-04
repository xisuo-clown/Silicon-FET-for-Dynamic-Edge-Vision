import numpy as np
from matplotlib import pyplot as plt


def img_show(name,if_fet=False,seq=1077):

    tags = ['Positive', 'Negative']
    # cmap = ['OrRd', 'BuPu']
    # cmap = 'magma'
    cmap = 'viridis'

    output_arr = np.load(name, allow_pickle=True)
    if if_fet:
        output_arr=output_arr[seq]
    else:
        output_arr=output_arr[0][0]
    # print(output_arr[0].shape)
    # output_arr = output_arr[0]
    print(output_arr.shape)
    # ########show a given frame, after stack, and augmentation
    # output_arr_stack = np.load(path_2, allow_pickle=True)
    # output_arr = output_arr_stack[n]
    # print(output_arr.shape)
    # label_list = np.load(path_1, allow_pickle=True)
    # label = label_list[n]
    # # #################################
    # ########show a given frame, after stack, remove the extra set,and augmentation
    # output_arr_stack = np.load(path_4, allow_pickle=True)
    # output_arr = output_arr_stack[n]
    # print(output_arr.shape)
    # label_list = np.load(path_3, allow_pickle=True)
    # label = label_list[n]
    # ######################display negative and positive###############
    fig, ax = plt.subplots(1, 2)
    # plt.suptitle("Test set {0}, Class {1}".format(a, label[n]))
    ax[0].imshow(output_arr[:, :, 0], cmap=cmap)
    ax[0].set_title(tags[0])
    ax[1].imshow(output_arr[:, :, 1], cmap=cmap)
    ax[1].set_title(tags[1])
    fig.tight_layout()
    # fig.colorbar
    # print(output_arr[:, :, 0]*1E5)
    fig.show()
    # ##########dispaly neg/positive end##############
    # import matplotlib.pyplot as plt
    plt.imshow(output_arr[:, :, 1], cmap=cmap)
    plt.colorbar(plt.imshow(output_arr[:, :, 1]), cmap=cmap)
    plt.show()
img_show("x_train.npy",True,0)
img_show("Aug_0.npy")