import math
import os.path
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from event_stream import init_event
from event_stream import polar_save_as_list
import numba as nb
from params_adjustment import calculate_match

c = 3E-4

# c = 1E-5
# the pulse interval (in pn vs id curve)
t_interval = 2E-4

# 1. n_clip: decide the length of each part (split the 6-second event stream into n_clip shorter event stream)
# (e.g. 2s event stream => n_clip = 3)
n_clip = 3
# 2. n_num: sum of divided parts(5=>split into 5 parts)!!!!!!!when change the n_num, the n_step may need to reset
# n_num = 5
n_num = 3
# 3. n_step: the step (the gap of the events number from the beginning of present
# part to the beginning of next part) (e.g. 6=>total events/6 as step length)
n_step = 6


class Dataset:
    def __init__(self):
        self.train_set_eve, self.test_set_eve = init_event()
        self.train_set_labels = self.train_set_eve.targets
        self.test_set_labels = self.test_set_eve.targets
        self.train_class_to_idx = self.train_set_eve.class_to_idx
        self.test_set_eve.class_to_idx = self.test_set_eve.class_to_idx
        self.ave_fre_count = self.cal_frequency_for_each_class()
        # self.ave_fre_count=self.cal_frequency_for_each_class_slide_window(3,6,3)
        # self.params = params

    def cal_frequency_for_each_class(self):
        frequency = defaultdict(list)
        for eve in self.train_set_eve:
            stream, label = eve
            frequency[label].append(len(tuple(stream["t"])) / (stream["t"][-1] - stream["t"][0]))
        ave_fre_count = []
        for fre in frequency:
            ave_fre_count.append(sum(frequency[fre]) / len(frequency[fre]))
        return ave_fre_count

    def cal_frequency_for_each_class_slide_window(self, window_num, step_ratio, size_ratio):
        frequency = defaultdict(list)
        for eve in self.train_set_eve:
            stream, label = eve
            freq = []
            for i in range(window_num):
                window_size = len(tuple(stream["t"])) // size_ratio
                window_step = len(tuple(stream["t"])) // step_ratio
                n_b = window_step * i
                n_e = n_b + window_size
                t = stream["t"][n_e] - stream["t"][n_b]
                n = window_size + 1
                freq.append(n / t)
            frequency[label] += freq

        ave_fre_count = []
        for fre in frequency:
            ave_fre_count.append(sum(frequency[fre]) / len(frequency[fre]))
            data = frequency[fre]
            mu = np.mean(data)  # 所有元素的均值
            sigma = np.std(data)  # 所有元素的标准差

            x_values = np.linspace(-5, 15, 1000)
            y_values = norm.pdf(x_values, mu, sigma)

            plt.figure(figsize=(8, 5))
            plt.plot(x_values, y_values, label=f'Global (μ={mu:.2f}, σ={sigma:.2f})', color='red')

            plt.xlabel("Value")
            plt.ylabel("Probability Density")
            plt.title("Gaussian Distribution for Entire 2D Array")
            plt.legend()
            plt.show()
        return ave_fre_count


# here we removed class 2 (other gestures)


class Datasample:
    def __init__(self, set_eve, params: calculate_match, title: str = "default"):
        self.title = title
        self.event, self.label = set_eve
        self.params = params
        self.id_pos, self.id_neg, self.id_pos_verse_t, self.id_neg_verse_t = self.current_generator()
        self.id_pos_max_idx, self.id_pos_min_idx = self.get_pos_idx()
        self.id_neg_max_idx, self.id_neg_min_idx = self.get_neg_idx()

    def get_pos_idx(self):
        # get max value of pos
        id_pos_max_idx = np.unravel_index(np.argmax(self.id_pos), self.id_pos.shape)

        # get min value of pos
        non_zero_image = self.id_pos[self.id_pos != 0]
        min_value = np.min(non_zero_image)
        id_pos_min_idx = np.where(self.id_pos == min_value)
        id_pos_min_idx = (id_pos_min_idx[0][0], id_pos_min_idx[1][0])

        return id_pos_max_idx, id_pos_min_idx

    def get_neg_idx(self):
        # get max value of neg
        id_neg_max_idx = np.unravel_index(np.argmax(self.id_neg), self.id_neg.shape)

        # get min value of neg
        non_zero_image = self.id_neg[self.id_neg != 0]
        min_value = np.min(non_zero_image)
        id_neg_min_idx = np.where(self.id_neg == min_value)
        id_neg_min_idx = (id_neg_min_idx[0][0], id_neg_min_idx[1][0])

        return id_neg_max_idx, id_neg_min_idx

    def event_stream_generator(self):
        event, label = self.event, self.label
        x0 = tuple(event["x"])
        y0 = tuple(event["y"])
        t = tuple(event["t"])
        p = tuple(event["p"])
        indexarr = index_arr()
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
            ne = n + int(len(t) / n_clip) - 1
            # print('n is {}'.format(n))
            # print('ne is {}'.format(ne))
            end_t = (t[ne] - t[n]) * 1E-6 * c
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
                        events_dict_neg[key].append((t[i] - t[n]) * 1E-6 * c)
                    else:
                        events_dict_pos[key].append((t[i] - t[n]) * 1E-6 * c)
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
                        # for l in events_dict[k]:
                        for l in events_dict[k]:
                            if l != events_dict[k][0]:
                                events_dict_time[k].append(l - l_0)
                            l_0 = l
                            events_dict_time[k].append(end_t - events_dict[k][-1])
            dict_pos_time[j] = events_dict_pos_time
            dict_neg_time[j] = events_dict_neg_time
        return dict_pos_time[0], dict_neg_time[0]

    def current_generator(self):
        from params_adjustment import calculate_match
        import numpy as np
        pos_temp_save = []
        neg_temp_save = []
        temp_save = [pos_temp_save, neg_temp_save]
        id_verse_t_save = [[], []]
        # y0, A1, A2, A3, t1, t2, t3, d_0, l_a, l_b, id_0 = transistor_3_exp()
        i_d = self.params
        dict_pos_time, dict_neg_time = self.event_stream_generator()
        dict_list = [dict_pos_time, dict_neg_time]
        output_arr = np.empty((2, 128, 128))
        output_arr_t_id = np.empty((2, 128, 128), dtype=object)
        indexarr = index_arr()
        output_id_verse_time = np.empty((2, 128, 128))

        # pos_events_dict = dict_pos[d]
        # neg_events_dict = dict_neg[d]
        for polar in range(2):
            temp_save[polar] = []
            id_verse_t_save[polar] = []
            events_dict = dict_list[polar]
            # print("event list is {}".format(events_dict))
            for i in range(128 * 128):
                if events_dict[i]:
                    tmp_id = []
                    id_last = i_d.d[0]
                    t = 0
                    # tmp_id.append((t, id_last))
                    for j in events_dict[i]:
                        y_0, a_1, a_2, a_3, t_1, t_2, t_3, d_, l_a, l_b = i_d.get_para(id_last)
                        t += j
                        # id_b,id_a=id_time_new(id_last, t, y_0, a_1, a_2, a_3, t_1, t_2, t_3, d_,l_a,l_b)
                        id_b = id_decay(id_last, j, y_0, a_1, a_2, a_3, t_1, t_2, t_3, d_)
                        id_a = id_pulse(id_b, l_a, l_b, i_d.id_th, i_d.id_0)

                        if j != events_dict[i][-1]:
                            id_last = id_a

                            tmp_id.append((t, id_b))
                            # tmp_id.append((t, id_a))
                        else:
                            id_last = id_b

                    if id_last - 15.2 > 0:
                        temp_save[polar].append(id_last - 15.2)

                        tmp_id.append((t, id_last - 15.2))

                    else:
                        temp_save[polar].append(0)

                        tmp_id.append((t, 0))

                    id_verse_t_save[polar].append(tmp_id)
                    # # >>>>>>>>>>??????????????????????
                    # if id_last - 15.2 < 0:
                    #     print(id_last)

                    # temp_save[polar].append(id_last)
                else:
                    temp_save[polar].append(0)

                    id_verse_t_save[polar].append([(0, 0)])

            for k in range(128):
                for m in range(128):
                    output_arr[polar, k, m] = temp_save[polar][int(indexarr[m][k])]
                    output_arr_t_id[polar, k, m] = id_verse_t_save[polar][int(indexarr[m][k])]
        return output_arr[0], output_arr[1], output_arr_t_id[0], output_arr_t_id[1]

    def plot_pic(self):
        title = self.title
        import event_stream
        import matplotlib.pyplot as plt
        tags = ['Positive', 'Negative']
        cmap = 'viridis'
        output_arr = np.array([self.id_pos, self.id_neg])
        # output_arr_stack = np.load(dataset_path + 'dataset_features_remove.npy', allow_pickle=True)
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
        plt.suptitle("{}".format(title))
        ax[0].imshow(output_arr[0, :, :], cmap=cmap)
        ax[0].set_title(tags[0])
        ax[1].imshow(output_arr[1, :, :], cmap=cmap)
        ax[1].set_title(tags[1])
        fig.tight_layout()
        # fig.colorbar
        # print(output_arr[:, :, 0]*1E5)
        fig.show()
        # ##########dispaly neg/positive end##############
        # import matplotlib.pyplot as plt

    def plt_pixel(self, m, n):
        try:
            x_neg, x_pos, y_neg, y_pos = self.id_verse_time_curve_generator(self.id_pos_verse_t[m][n],
                                                                            self.id_neg_verse_t[m][n])

            fig, ax = plt.subplots(2, 1)
            plt.suptitle("The id verse time plot of row {} col {}".format(m, n))
            ax[0].plot(x_pos, y_pos, label="positive curve")
            ax[0].set_title("positive")
            ax[1].plot(x_neg, y_neg, label="negative curve")
            ax[1].set_title("negative")
            plt.tight_layout()  # 自动调整布局，防止重叠
            plt.show()

        except IndexError:
            print("Index out of range")
            raise

    def id_verse_time_curve_generator(self, pixel_in_pos, pixel_in_neg):
        intervals = 50
        i_d = self.params
        pixel_in_pos=np.array(pixel_in_pos)
        pixel_in_neg=np.array(pixel_in_neg)
        x_ori = [pixel_in_pos[:, 0], pixel_in_neg[:, 0]]
        y_ori = [pixel_in_pos[:, 1], pixel_in_neg[:, 1]]
        x=[[],[]]
        y=[[],[]]
        id_last=i_d.d[0]
        t_last=0
        for i in range(2):
            for j in range(len(x_ori[i])):
                x[i].append(t_last)
                y[i].append(id_last)
                y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_, l_a, l_b = i_d.get_para(id_last)
                t=np.linspace(t_last, x_ori[i][j], intervals).tolist()
                v=[id_decay(id_last,x-t_last,y_0,A_1, A_2, A_3, t_1, t_2, t_3, d_) for x in t]
                x[i].extend(t)
                y[i].extend(v)
                id_last=id_pulse(y_ori[i][j],l_a,l_b,i_d.id_th,i_d.id_0)
                t_last=x_ori[i][j]
                if j==len(x_ori[i])-1:
                    x[i].append(t_last)
                    y[i].append(y_ori[i][j])


            id_last = i_d.d[0]
            t_last = 0




        return x[0],x[1],y[0],y[1]

    def plot_param_curve(self):
        import matplotlib.pyplot as plt

        x_begin = 0
        intervals = 10000
        fig, ax = plt.subplots(2, 3)
        plt.suptitle("exp function curve")
        for i in range(3):
            x, y_e1, y_e2, y_e3, y_sum = self.exp_curve_generator(i, intervals, x_begin)
            ax[0][i].plot(x, y_e1, label="e1", color='r')
            ax[0][i].plot(x, y_e2, label="e2", color='b')
            ax[0][i].plot(x, y_e3, label="e3", color='g')
            ax[1][i].plot(x, y_sum, label="sum", color='y')
        plt.tight_layout()  # 自动调整布局，防止重叠
        plt.show()

    def exp_curve_generator(self, i, intervals, x_begin):
        y0 = self.params.y0[i]
        A1 = self.params.A1[i]
        t1 = self.params.t1[i]
        A2 = self.params.A2[i]
        t2 = self.params.t2[i]
        A3 = self.params.A3[i]
        t3 = self.params.t3[i]
        x_end = min([t1, t2, t3]) * 5
        x = np.linspace(x_begin, x_end, intervals).tolist()
        y_e1 = [A1 * math.exp(-t / t1) for t in x]
        y_e2 = [A2 * math.exp(-t / t2) for t in x]
        y_e3 = [A3 * math.exp(-t / t3) for t in x]
        y_sum = [A1 * math.exp(-t / t1) + A2 * math.exp(-t / t2) + A2 * math.exp(-t / t3) + y0 for t in x]
        return x, y_e1, y_e2, y_e3, y_sum

    def plot_comparative_curve(self, d2, m, n):
        x_begin = 0
        x_end = 0.1
        intervals = 10000
        fig, ax = plt.subplots(3, 1)
        for i in range(3):
            x_1, y_e1_1, y_e2_1, y_e3_1, y_sum_1 = self.exp_curve_generator(i, intervals, x_begin)
            ax[i].plot(x_1, y_sum_1, label="exp full {}".format(self.title), color='r')

            x_2, y_e1_2, y_e2_2, y_e3_2, y_sum_2 = d2.exp_curve_generator(i, intervals, x_begin)
            ax[i].plot(x_2, y_sum_2, label="exp full {}".format(d2.title), color='g')
            plt.tight_layout()

            # ax[i].plot(x_1, y_e1_1, label="exp 1 {}".format(self.title), color='r')
            # ax[i].plot(x_2, y_e1_2, label="exp 1 {}".format(d2.title), color='g')
            # plt.tight_layout()
            # plt.show()  # 调整布局
            #
            # ax[i].plot(x_1, y_e2_1, label="exp 2 {}".format(self.title), color='r')
            # ax[i].plot(x_2, y_e2_2, label="exp 2 {}".format(d2.title), color='g')
            # plt.tight_layout()
            # plt.show()  # 调整布局
            #
            # ax[i].plot(x_1, y_e3_1, label="exp 3 {}".format(self.title), color='r')
            # ax[i].plot(x_2, y_e3_2, label="exp 3 {}".format(d2.title), color='g')
            # plt.tight_layout()
            # plt.show()  # 调整布局
        plt.show()  # 调整布局

        fig, ax = plt.subplots(1, 1)
        plt.suptitle("comparative curve of {} and {} of row {} col {}".format(self.title, d2.title,m,n))
        x_pos_1, x_neg_1, y_pos_1, y_neg_1 = self.id_verse_time_curve_generator(self.id_pos_verse_t[m][n],
                                                                                self.id_neg_verse_t[m][n])
        x_pos_2, x_neg_2, y_pos_2, y_neg_2 = d2.id_verse_time_curve_generator(d2.id_pos_verse_t[m][n],
                                                                                d2.id_neg_verse_t[m][n])
        x_pos_edge_1,y_pos_edge_1=find_duplicate_x_and_y(x_pos_1, y_pos_1)
        x_pos_edge_2,y_pos_edge_2=find_duplicate_x_and_y(x_pos_2, y_pos_2)
        x_neg_edge_1,y_neg_edge_1=find_duplicate_x_and_y(x_neg_1, y_neg_1)
        x_neg_edge_2,y_neg_edge_2=find_duplicate_x_and_y(x_neg_2, y_neg_2)

        ax.plot(x_pos_1, y_pos_1, label="pos id verse time of row {} col {} pixel {}".format(m, n, self.title),
                   color='r')
        ax.plot(x_pos_2, y_pos_2, label="pos id verse time of row {} col {} pixel {}".format(m, n, d2.title),
                   color='g'
                   )
        ax.scatter(x_pos_edge_1, y_pos_edge_1,color='r')
        ax.scatter(x_pos_edge_2, y_pos_edge_2,color='g')
        plt.legend()
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(1, 1)
        plt.suptitle("comparative curve of {} and {}".format(self.title, d2.title))
        ax.plot(x_neg_1, y_neg_1, label="neg id verse time of row {} col {} pixel {}".format(m, n, self.title),
                   color='r')
        ax.plot(x_neg_2, y_neg_2, label="neg id verse time of row {} col {} pixel {}".format(m, n, d2.title),
                   color='g')
        ax.scatter(x_neg_edge_1, y_neg_edge_1,color='r')
        ax.scatter(x_neg_edge_2, y_neg_edge_2,color='g')
        plt.legend()
        plt.tight_layout()
        plt.show()


def index_arr():
    import numpy as np
    h = 0
    indexarr = np.zeros((128, 128))
    for i in range(128):
        for j in range(128):
            indexarr[i][j] = h
            h += 1
    return indexarr


@nb.jit(nopython=True)
def id_time_new(i_last, t, y0, A1, A2, A3, t1, t2, t3, d, l_a, l_b):
    id_b = (i_last / d) * (A1 * math.exp(-t / t1) + A2 * math.exp(-t / t2) + A3 * math.exp(-t / t3) + y0)
    # print(id_vs_time)
    id_a = l_a * id_b + l_b
    if id_a > 30.5:
        id_a = 30.5
    return id_b, id_a


@nb.jit(nopython=True)
def id_pulse(i_last, l_a, l_b, id_th, id_th_0):
    id_a = l_a * i_last + l_b
    if id_a > id_th:
        id_a = id_th
    if id_th_0 > 0 and id_a < id_th_0:
        id_a = id_th_0
    return id_a


@nb.jit(nopython=True)
def id_decay(i_last, t, y0, a1, a2, a3, t1, t2, t3, d):
    # if t==0:
    #     return i_last
    return (i_last / d) * (a1 * math.exp(-t / t1) + a2 * math.exp(-t / t2) + a3 * math.exp(-t / t3) + y0)


def find_duplicate_x_and_y(x, y):
    counter = defaultdict(list)
    [counter[v].append(i) for i, v in enumerate(x)]
    counter = {k: v for k, v in counter.items() if len(v) > 1}
    idx = [i for value in counter.values() for i in value]
    return [x[i] for i in idx], [y[i] for i in idx]
