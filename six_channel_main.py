import os

if __name__ == '__main__':
    #gpu set
    from tools import find_gpu_with_min_usage
    from params_adjustment import find_path
    try:
        find_gpu_with_min_usage()
    except Exception as e:
        print(e)

    suffix = "ltsm_resnet_3clips"
    root_dir, train_save_path, test_save_path = find_path(suffix)
    # data generation
    from params_adjustment import calculate_match, find_path

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
    para = calculate_match(y0, A1, t1, A2, t2, A3, t3, d, a, b)
    from six_channel_event_frame import polarity_process_transistor_conditions,gen_augmentation_frame
    if not os.path.exists(os.path.join(train_save_path, "Aug_dataset_features.npy")):
        polarity_process_transistor_conditions(True,para,suffix)
        polarity_process_transistor_conditions(False, para, suffix)
        gen_augmentation_frame(suffix)

    #training

    from six_channel_resnet import hyper_tuner
    from datetime import datetime
    while True:

        name = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_name=os.path.join(test_save_path,"test_results_3_resnet")
        hyper_tuner(True,name,dir_name,suffix,True)