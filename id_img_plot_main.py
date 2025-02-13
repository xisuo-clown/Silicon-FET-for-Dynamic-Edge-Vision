import os.path

import numpy as np
from tqdm import tqdm

import id_verse_time
from params_adjustment import calculate_match


def find_biggest_change(data1, data2):
    diff = np.abs(data1.id_pos - data2.id_pos)
    max_index = np.unravel_index(np.argmax(diff), diff.shape)
    return max_index


if __name__ == '__main__':
    y0 = [20.06402, 20.31625, 20.55575]
    A1 = [6.30373, 6.65462, 9.10203]
    t1 = [1.49676E-5, 1.93912E-5, 2.84333E-5]
    A2 = [4.72692, 4.61784, 2.19681]
    t2 = [8.72838E-5, 1.09493E-4, 2.77744E-4]
    A3 = [0.5666, 0.45408, 0.59678]
    t3 = [0.01394, 0.12556, 1.29371]
    d = [28.81179, 29.65465, 30.59423]
    a = 0.89091
    b = 6.78201
    id_0 = 0
    id_th = 30.5
    para_before_tune = calculate_match(y0, A1, t1, A2, t2, A3, t3, d, a, b, id_0, id_th)

    # y0 = [20.01978, 20.42166, 20.75204]
    # A1 = [5.15863, 7.37592, 7.96576]
    # t1 = [7.53665E-5, 2.2508E-5, 2.60293E-5]
    # A2 = [5.13078, 2.82181, 2.64961]
    # t2 = [1.32347E-5, 1.52147E-4, 1.96754E-4]
    # A3 = [0.35981, 0.35254, 0.38272]
    # t3 = [0.02205, 0.31306, 0.25639]
    # d = [27.9274791, 28.8670617, 29.73755545]
    # a = 0.68193
    # b = 11.48977
    y0 = [20.01978, 20.42166, 20.75204]
    A1 = [5.15863, 7.37592, 7.96576]
    t1 = [7.53665E-5, 2.2508E-5, 2.60293E-5]
    A2 = [5.13078, 2.82181, 2.64961]
    t2 = [1.32347E-5, 1.52147E-4, 1.96754E-4]
    A3 = [0.35981, 0.35254, 0.38272]
    t3 = [0.02205, 0.31306, 0.25639]
    d = [27.9274791, 28.8670617, 29.73755545]
    a = 0.68193
    b = 11.48977
    id_0 = d[0]
    id_th = d[2]
    para_after_tune = calculate_match(y0, A1, t1, A2, t2, A3, t3, d, a, b, id_0, id_th)

    y0 = [20.01978, 20.42166, 20.75204]
    A1 = [5.15863, 7.37592, 7.96576]
    t1 = [7.53665E-5, 2.2508E-5, 2.60293E-5]
    A2 = [5.13078, 2.82181, 2.64961]
    t2 = [1.32347E-5, 1.52147E-4, 1.96754E-4]
    A3 = [0.35981, 0.35254, 0.38272]
    t3 = [0.02205, 0.31306, 0.25639]
    d = [27.9274791, 28.8670617, 29.73755545]
    a = 0.68193
    b = 11.48977
    id_0 = 0
    id_th = d[2]
    para_after_tune_without_lower_current_limit = calculate_match(y0, A1, t1, A2, t2, A3, t3, d, a, b, id_0, id_th)

    idx = 999

    dataset = id_verse_time.Dataset()
    data_before_tune = id_verse_time.Datasample(dataset.train_set_eve[idx], para_before_tune, "before tune")
    data_after_tune = id_verse_time.Datasample(dataset.train_set_eve[idx], para_after_tune, "after tune")
    data_after_tune_without_lcl = id_verse_time.Datasample(dataset.train_set_eve[idx],
                                                           para_after_tune_without_lower_current_limit,
                                                           "after tune no limits")
    biggest_change_idx = find_biggest_change(data_before_tune, data_after_tune)

    data_before_tune.plt_pixel(biggest_change_idx[0], biggest_change_idx[1])
    data_after_tune.plt_pixel(biggest_change_idx[0], biggest_change_idx[1])
    data_after_tune_without_lcl.plt_pixel(biggest_change_idx[0], biggest_change_idx[1])

    data_before_tune.plot_pic()
    data_after_tune.plot_pic()
    data_after_tune_without_lcl.plot_pic()
    data_before_tune.plot_comparative_curve(data_after_tune, biggest_change_idx[0], biggest_change_idx[1])
    data_before_tune.plot_comparative_curve(data_after_tune_without_lcl, biggest_change_idx[0], biggest_change_idx[1])