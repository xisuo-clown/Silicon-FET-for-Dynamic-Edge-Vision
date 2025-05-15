import math
import os.path
from collections import defaultdict
from datetime import time

import numpy as np
from matplotlib import pyplot as plt

from event_stream import init_event
from event_stream import polar_save_as_list
import numba as nb
from params_adjustment import calculate_match
from event_stream import c


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
        print("Initializing Dataset")
        self.train_set_eve, self.test_set_eve = init_event()
        self.train_set_labels = self.train_set_eve.targets
        self.test_set_labels = self.test_set_eve.targets
        self.train_class_to_idx = self.train_set_eve.class_to_idx
        self.test_set_eve.class_to_idx = self.test_set_eve.class_to_idx
        self.fre_count = self.cal_frequency_for_each_class()
        self.label = self.get_label()

        print()
        # self.ave_fre_count=self.cal_frequency_for_each_class_slide_window(3,6,3)
        # self.params = params

    def get_label(self):
        label = defaultdict(list)
        for i, v in enumerate(self.train_set_eve.targets):
            label[v].append(i)
        return label

    def cal_frequency_for_each_class(self):
        frequency = [[] for _ in range(10)]
        for eve in self.train_set_eve:
            stream, label = eve
            if label > 2:
                frequency[label - 1].append(len(tuple(stream["t"])) * 1e6 / (stream["t"][-1] - stream["t"][0]))
            elif label == 2:
                continue
            elif label < 2:
                frequency[label].append(len(tuple(stream["t"])) * 1e6 / (stream["t"][-1] - stream["t"][0]))

        return frequency

    def plot_3d(self, n, only_cal=False):
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        event, label = self.train_set_eve[n]
        if label > 2:
            label -= 1
        x = event["x"]
        y = event["y"]
        t = event["t"]
        p = event["p"]
        mask = (p == 1)
        x = x[mask]
        y = y[mask]
        t = t[mask]

        l = 0
        for idx, t1 in enumerate(t):
            if t1 - t[0] > 2e6:
                l = idx
                break
        if l == 0:
            l = len(t) - 1
        # l=len(t)//3

        x_pos = x[0:l + 1]
        y_pos = y[0:l + 1]
        t_pos = t[0:l + 1]
        t_pos -= t_pos[0]
        data_num = len(x_pos)
        # 生成示例数据

        # 创建 3D 图形

        for s_value in [0.02 * (i + 1) for i in range(1)]:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(t_pos, x_pos, y_pos, c=t_pos, cmap="viridis", marker='s', s=s_value)

            # 设置坐标轴比例和标签
            ax.set_box_aspect([2, 1, 1])
            ax.set_xlabel('t')
            ax.set_ylabel('x')
            ax.set_zlabel('y')
            ax.text(np.max(t_pos) / 2, -64,
                    -64, f"Data Points: {data_num}",
                    fontsize=12, color="blue", ha="left", va="bottom")
            print("max t: ", max(t_pos))
            print("min t: ", min(t_pos))
            plt.title("3D Scatter Plot of number {} label {} with s={}".format(n, label, s_value))
            plt.show()
        return x_pos, y_pos, t_pos


# here we removed class 2 (other gestures)


class Datasample:
    def __init__(self, set_eve, params: calculate_match, title: str = "default"):
        self.title = title
        self.event, self.label = set_eve
        self.params = params
        self.id_pos, self.id_neg, self.id_pos_verse_t, self.id_neg_verse_t = self.current_generator()
        self.id_pos_max_idx, self.id_pos_min_idx = self.get_pos_idx()
        self.id_neg_max_idx, self.id_neg_min_idx = self.get_neg_idx()

    def draw_scatter_of_cur_verse_pulse(self,t1,t2,p):
        import numpy as np
        import matplotlib.pyplot as plt
        from scipy.stats import spearmanr
        fre_1 = self.get_number_of_pulses_within_time(t1, p)
        cur_1 = self.get_final_current_within_time(t1, p)
        fre_2 = self.get_number_of_pulses_within_time(t2, p)
        cur_2 = self.get_final_current_within_time(t2, p)
        fre_1 = np.array(fre_1, dtype=float).flatten()
        cur_1 = np.array(cur_1, dtype=float).flatten()
        fre_2 = np.array(fre_2, dtype=float).flatten()
        cur_2 = np.array(cur_2, dtype=float).flatten()

        fre_set_1=set(fre_1)
        fre_set_2=set(fre_2)
        std_1,count_1=0,0
        std_2,count_2=0,0
        for k in fre_set_1:
            std_1+=np.std(cur_1[fre_1==k])
            count_1+=1
        for k in fre_set_2:
            std_2+=np.std(cur_2[fre_2==k])
            count_2+=1
        std_1/=count_1
        std_2/=count_2

        cor_1 = std_1
        cor_2 = std_2
        # 打印相关系数
        print("Spearman Correlation 1:", cor_1)
        print("Spearman Correlation 2:", cor_2)

        # 创建散点图
        plt.figure(figsize=(10, 6))

        # 绘制 fre_1 vs cur_1 散点图
        plt.subplot(1, 2, 1)
        plt.scatter(fre_1, cur_1, color='blue', label=f'Uncertainty Value: {cor_1:.4f}')
        plt.title('0.5-second clip')
        plt.xlabel('number of events')
        plt.ylabel('current')
        plt.legend()

        # 绘制 fre_2 vs cur_2 散点图
        plt.subplot(1, 2, 2)
        plt.scatter(fre_2, cur_2, color='green', label=f'Uncertainty Value: {cor_2:.4f}')
        plt.title('2-second clip')
        plt.xlabel('number of events')
        plt.ylabel('current')
        plt.legend()

        # 在图上标注相关系数
        # 将相关系数标注在每个图的右上角
        plt.subplot(1, 2, 1)


        plt.subplot(1, 2, 2)


        # 展示所有子图
        plt.tight_layout()
        plt.show()

    def get_number_of_pulses_within_time(self, t_, p_):
        event_arr = [[0 for _ in range(128)] for _ in range(128)]
        x = self.event["x"]
        y = self.event["y"]
        t = self.event["t"]
        p = self.event["p"]
        for i in range(len(t)):
            if t[i] - t[0] < t_*1e6:
                if p[i] == p_:
                    event_arr[x[i]][y[i]] += 1
            else:
                break
        return event_arr



    def get_final_current_within_time(self, t_, p_):
        event_arr = [[[] for _ in range(128)] for _ in range(128)]
        x = self.event["x"]
        y = self.event["y"]
        t = self.event["t"]
        p = self.event["p"]
        for i in range(len(t)):
            if t[i] - t[0] < t_*1e6:
                if p[i] == p_:
                    event_arr[x[i]][y[i]].append((t[i]-t[0])*c*1e-6)
            else:
                break
        event_arr_t_dif = [[[] for _ in range(128)] for _ in range(128)]
        cur_arr= [[0 for _ in range(128)] for _ in range(128)]
        for i in range(len(event_arr)):
            for j in range(len(event_arr[i])):
                if event_arr[i][j]:
                    l=event_arr[i][j][0]
                    for k in event_arr[i][j]:
                        if k!=l:
                            event_arr_t_dif[i][j].append(k-l)
                        l=k
                    event_arr_t_dif[i][j].append(t_-event_arr[i][j][-1])
                    i_last=self.params.d[0]
                    for t in event_arr_t_dif[i][j]:
                        y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_, l_a, l_b=self.params.get_para(i_last)
                        id_b, id_a=id_time_new(i_last, t, y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_, l_a, l_b)
                        if t==event_arr_t_dif[i][j][-1]:
                            i_last=max(0,id_b-15.2)
                        else:
                            i_last=id_a
                    cur_arr[i][j]=i_last
                else:
                    i_last = 0
                    cur_arr[i][j]=i_last
        return cur_arr


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
        print(output_arr.shape)
        fig, ax = plt.subplots(1, 2)
        plt.suptitle("{}".format(title))
        ax[0].imshow(output_arr[0, :, :], cmap=cmap)
        ax[0].set_title(tags[0])
        ax[1].imshow(output_arr[1, :, :], cmap=cmap)
        ax[1].set_title(tags[1])
        brightness = np.mean(output_arr, axis=(0, 1))
        contrast = np.std(output_arr, axis=(0, 1)) / brightness
        fig.text(0.05, 0.9, f"Brightness: {brightness[0]:.2f}", fontsize=12, color='black', ha='left', va='top')
        fig.text(0.55, 0.9, f"Brightness: {brightness[1]:.2f}", fontsize=12, color='black', ha='left', va='top')
        fig.text(0.05, 0.85, f"Contrast: {contrast[0]:.2f}", fontsize=12, color='black', ha='left', va='top')
        fig.text(0.55, 0.85, f"Contrast: {contrast[1]:.2f}", fontsize=12, color='black', ha='left', va='top')
        fig.tight_layout()
        fig.show()
        # ##########dispaly neg/positive end##############
        # import matplotlib.pyplot as plt

    def plt_pixel(self, m, n):
        try:
            x_neg, x_pos, y_neg, y_pos = self.id_verse_time_curve_generator(self.id_pos_verse_t[m][n],
                                                                            self.id_neg_verse_t[m][n])
            cmap = 'viridis'
            fig, ax = plt.subplots(1, 1)
            plt.suptitle("The id verse time plot of row {} col {}".format(n, m))
            ax.plot(x_pos, y_pos, label="positive curve")
            ax.set_title("positive")
            x_pos_edge_1, y_pos_edge_1 = find_duplicate_x_and_y(x_pos, y_pos)
            ax.scatter(x_pos_edge_1, y_pos_edge_1, c='r')
            plt.tight_layout()  # 自动调整布局，防止重叠
            plt.show()

            fig, ax = plt.subplots(1, 1)
            output_arr = np.array([self.id_pos, self.id_neg])
            ax.imshow(output_arr[0, :, :], cmap=cmap)
            ax.set_title("positive")
            # ax[1].plot(x_neg, y_neg, label="negative curve")
            # ax[1].set_title("negative")
            length = 10
            ax.plot([m - length, m + length], [n, n], color='white', linewidth=2)  # 水平线
            ax.plot([m, m], [n - length, n + length], color='white', linewidth=2)  # 垂直线

            plt.tight_layout()  # 自动调整布局，防止重叠
            plt.show()

        except IndexError:
            print("Index out of range")
            raise

    def id_verse_time_curve_generator(self, pixel_in_pos, pixel_in_neg):
        intervals = 500
        i_d = self.params
        pixel_in_pos = np.array(pixel_in_pos)
        pixel_in_neg = np.array(pixel_in_neg)
        x_ori = [pixel_in_pos[:, 0], pixel_in_neg[:, 0]]
        y_ori = [pixel_in_pos[:, 1], pixel_in_neg[:, 1]]
        x = [[], []]
        y = [[], []]
        id_last = i_d.d[0]
        t_last = 0
        for i in range(2):
            for j in range(len(x_ori[i])):
                x[i].append(t_last)
                y[i].append(id_last)
                y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_, l_a, l_b = i_d.get_para(id_last)
                t = np.linspace(t_last, x_ori[i][j], intervals).tolist()
                v = [id_decay(id_last, x - t_last, y_0, A_1, A_2, A_3, t_1, t_2, t_3, d_) for x in t]
                x[i].extend(t)
                y[i].extend(v)
                id_last = id_pulse(y_ori[i][j], l_a, l_b, i_d.id_th, i_d.id_0)
                t_last = x_ori[i][j]
                if j == len(x_ori[i]) - 1:
                    x[i].append(t_last)
                    y[i].append(y_ori[i][j])

            id_last = i_d.d[0]
            t_last = 0

        return x[0], x[1], y[0], y[1]

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
        d = self.params.d[i]
        x_end = min([t1, t2, t3]) * 5
        x = np.linspace(x_begin, x_end, intervals).tolist()
        y_e1 = [A1 * math.exp(-t / t1) for t in x]
        y_e2 = [A2 * math.exp(-t / t2) for t in x]
        y_e3 = [A3 * math.exp(-t / t3) for t in x]
        y_sum = [(A1 * math.exp(-t / t1) + A2 * math.exp(-t / t2) + A2 * math.exp(-t / t3) + y0) / d for t in x]
        return x, y_e1, y_e2, y_e3, y_sum

    def plot_comparative_curve(self, d2, m, n):
        x_begin = 0
        x_end = 0.1
        intervals = 10000
        fig, ax = plt.subplots(3, 1)
        for i in range(3):
            x_1, y_e1_1, y_e2_1, y_e3_1, y_sum_1 = self.exp_curve_generator(i, intervals, x_begin)
            ax[i].plot(x_1, y_sum_1, label="exp full {}".format(self.title), color='r')
            plt.title("a1,t1,a2,t2,a3,t3")
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
        plt.legend()
        plt.show()  # 调整布局

        fig, ax = plt.subplots(1, 1)

        x = np.linspace(0, 35, 350)
        y_self = x * self.params.a + self.params.b
        y_d = x * d2.params.a + d2.params.b
        ax.plot(x, y_self, label="{}".format(self.title), color='r')
        ax.plot(x, y_d, label="{}".format(d2.title), color='g')
        plt.title("ax+b")
        plt.legend()
        plt.show()

        fig, ax = plt.subplots(1, 1)
        plt.suptitle("comparative curve of {} and {} of row {} col {}".format(self.title, d2.title, m, n))
        x_pos_1, x_neg_1, y_pos_1, y_neg_1 = self.id_verse_time_curve_generator(self.id_pos_verse_t[m][n],
                                                                                self.id_neg_verse_t[m][n])
        x_pos_2, x_neg_2, y_pos_2, y_neg_2 = d2.id_verse_time_curve_generator(d2.id_pos_verse_t[m][n],
                                                                              d2.id_neg_verse_t[m][n])
        x_pos_edge_1, y_pos_edge_1 = find_duplicate_x_and_y(x_pos_1, y_pos_1)
        x_pos_edge_2, y_pos_edge_2 = find_duplicate_x_and_y(x_pos_2, y_pos_2)
        x_neg_edge_1, y_neg_edge_1 = find_duplicate_x_and_y(x_neg_1, y_neg_1)
        x_neg_edge_2, y_neg_edge_2 = find_duplicate_x_and_y(x_neg_2, y_neg_2)

        ax.plot(x_pos_1, y_pos_1, label="pos id verse time of row {} col {} pixel {}".format(m, n, self.title),
                color='r')
        ax.plot(x_pos_2, y_pos_2, label="pos id verse time of row {} col {} pixel {}".format(m, n, d2.title),
                color='g'
                )
        ax.scatter(x_pos_edge_1, y_pos_edge_1, color='r')
        ax.scatter(x_pos_edge_2, y_pos_edge_2, color='g')
        plt.legend()
        plt.tight_layout()
        plt.show()

        fig, ax = plt.subplots(1, 1)
        plt.suptitle("comparative curve of {} and {}".format(self.title, d2.title))
        ax.plot(x_neg_1, y_neg_1, label="neg id verse time of row {} col {} pixel {}".format(m, n, self.title),
                color='r')
        ax.plot(x_neg_2, y_neg_2, label="neg id verse time of row {} col {} pixel {}".format(m, n, d2.title),
                color='g')
        ax.scatter(x_neg_edge_1, y_neg_edge_1, color='r')
        ax.scatter(x_neg_edge_2, y_neg_edge_2, color='g')
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
