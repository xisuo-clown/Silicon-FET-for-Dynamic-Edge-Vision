import numpy as np
from matplotlib import pyplot as plt




def current_weights_test(seq_map):
    from params_adjustment import gen_current
    current_map = np.moveaxis(seq_map, 0, -1)
    return np.apply_along_axis(gen_current, axis=-1, arr=current_map)

def show_compare(seq_map):
    img_show(seq_map)
    img_show(current_weights_test(seq_map))

def img_show(content):
    if np.ndim(content) == 3:
        tags = "after_gen_current"
        # cmap = ['OrRd', 'BuPu']
        # cmap = 'magma'
        cmap = 'viridis'
        plt.imshow(content[:, :, 0], cmap=cmap)
        plt.colorbar(plt.imshow(content[:, :, 0]), cmap=cmap)
        plt.title(tags)
        plt.show()
    else:
        tags = ['Positive', 'Negative']
        # cmap = ['OrRd', 'BuPu']
        # cmap = 'magma'
        cmap = 'viridis'
        fig, ax = plt.subplots(2, 3)
        count=0
        for c in content:
            count_1=count//3
            count_2=count%3
            # plt.suptitle("Test set {0}, Class {1}".format(a, label[n]))
            ax[count_1][count_2].imshow(c[:, :, 0], cmap=cmap)
            ax[count_1][count_2].set_title(tags[0])
            count+=1


        fig.tight_layout()
        # fig.colorbar
        # print(output_arr[:, :, 0]*1E5)
        fig.show()


def img_show_by_name(name,if_fet=False,seq=1077):



    output_arr = np.load(name, allow_pickle=True)
    if if_fet:
        output_arr=output_arr[seq]
    else:
        output_arr=output_arr[0][0]
    # print(output_arr[0].shape)
    # output_arr = output_arr[0]
    print(output_arr.shape)
    img_show(output_arr)
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

for i in range(1000):
    name="x_train_10_3e-05.npy"
    test=np.load(name)[i]
    img_show(test)