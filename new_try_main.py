from new_try_for_data_gen import data_gen_test

if __name__ == '__main__':
    print("this is the beginning of the main function")
    from params_adjustment import calculate_match
    import tensorflow as tf
    from tools import find_gpu_with_min_usage
    try:
        find_gpu_with_min_usage()
    except Exception as e:
        print(e)
    # if gpus:
    #     try:
    #         for gpu in gpus:
    #             tf.config.experimental.set_memory_growth(gpu, True)
    #     except RuntimeError as e:
    #         print(e)

    # y0 = [20.06402, 20.31625, 20.55575]
    # A1 = [6.30373, 6.65462, 9.10203]
    # t1 = [1.49676E-5, 1.93912E-5, 2.84333E-5]
    # A2 = [4.72692, 4.61784, 2.19681]
    # t2 = [8.72838E-5, 1.09493E-4, 2.77744E-4]
    # A3 = [0.5666, 0.45408, 0.59678]
    # t3 = [0.01394, 0.12556, 1.29371]
    # d = [28.81179, 29.65465, 30.59423]
    # a = 0.89091
    # b = 6.78201
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

    suffix = ["no_tune_6clips", "tune_for_all_new_10", "tune_for_eight_new_10", "tune_for_nine_new_10"]
    tune_choice = [
        [False, False, False, False, False, False, False, False, False, False, False],
        [True, True, True, True, True, True, True, True, True, True, True],
        [False, False, False, False, False, False, False, False, False, True, False],
        [False, False, False, False, False, False, False, False, False, False, True]]

    # data_gen_test(para_before_tune=para_before_tune, para_after_tune=para_after_tune,
    #                                tune_choice=tune_choice[0], suffix=suffix[0], mode=False)
    data_gen_test(para_before_tune=para_before_tune, para_after_tune=para_after_tune,
                  tune_choice=tune_choice[0], suffix=suffix[0],results_path="resnet_num_{}".format(6),mode=False)

