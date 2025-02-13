



if __name__ == '__main__':
    import gaussian_distribution as g
    import params_adjustment as ap
    from tools import find_gpu_with_min_usage
    import numpy as np
    from params_adjustment import calculate_match

    find_gpu_with_min_usage()
    # ts=np.load("/usr1/home/s124mdg41_03/Integrated_package/DvsGes/event_array/train_set_eve_tune_1/Aug_dataset_labels_remove.npy")
    # params before tuning(commented)
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
    para_before_tune = calculate_match(y0, A1, t1, A2, t2, A3, t3, d, a, b)
    # params being adjusted (commented)
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

    # params after tuning(commented)
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


    para_after_tune = calculate_match(y0, A1, t1, A2, t2, A3, t3, d, a, b)
    # choosing which set to tune
    suffix=["tune_for_eight","tune_for_all","no_tune"]
    tune_choice = [[False, False, False, False, False, False, False, False, False, True, False],
                   [True, True, True, True, True, True, True, True, True, True, True],
                   [False, False, False, False, False, False, False, False, False, False, False]]
    from event_stream import polarity_process_transistor_conditions
    for i in range(len(suffix)):
        ap.dataset_generator_and_training(para_before_tune=para_before_tune, para_after_tune=para_after_tune,
                                      tune_choice=tune_choice[i],
                                      suffix=suffix[i], if_only_train=False)
    # ap.dataset_generator_and_training(
    #                                   suffix="eve_tune_1", if_only_train=False)

    # import Visualization as vis
    # # import improvement_tuning as it
    # # it.add_polarity_process_transistor(8,True,False)
    # vis.img_show(True, 1)
    # vis.img_show(True,980)
    # vis.img_show(True,980,True)
    # g.analyze_accuracy_distribution("results",False)
    # g.analyze_accuracy_distribution("results", True)
    # g.analyze_accuracy_distribution("dec_results", False)
    # g.analyze_accuracy_distribution("dec_results", True)
    # import twice_event_stream as t
    #
    # print(t.destination_dir())
