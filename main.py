import DVS_Animals

if __name__ == '__main__':
    import event_stream

    #
    # event_stream.polarity_process_transistor_match(train=True)
    # event_stream.polarity_process_transistor_match(train=False)

    # #
    # # event_stream.polarity_process_transistor_cal(train=True)
    # # event_stream.polarity_process_transistor_cal(train=False)

    # event_stream.polarity_process_transistor_conditions(train=True)
    # event_stream.polarity_process_transistor_conditions(train=False)

    #
    # #
    import frames_processing
    # #
    # # # #
    # # # # # # origin data
    # frames_processing.gen_stack_frame(Aug=False)
    # frames_processing.polar_remove_set7(aug=False)
    # # #
    # # #
    # #
    # # # # utilize augmented data
    # frames_processing.gen_augmentation_frame()
    # frames_processing.polar_remove_set7(aug=True)
    # #
    # #
    # # # ######################
    import improvement_tuning
    # 0-9 total 10 classes
    #
    # improvement_tuning.add_polarity_process_transistor(8, train=True, calculate=False)
    # improvement_tuning.add_polarity_process_transistor(8, train=False,calculate=False)
    # improvement_tuning.gen_augmentation_frame(8)

    # improvement_tuning.gen_tuned_stack_frame(9, tune=True)

    # # readout model
    import resnet_10
    # resnet_10.hyper_tuner(aug=True, tune=False)
    # resnet_10.hyper_tuner(aug=True, tune=True)

    import Logistic_Regression
    # Logistic_Regression.logistic_regression()
    # # # # # ########################
    from Visualization import img_show, img_show_tuning, img_show_before_tuning
    #     n = 0
    # img_show(if_train=True, a=1)
    #     # for i in range (6):
    #     #     a = n * 6 - 1 + i
    #     #     img_show(if_train=True, a=a)
    #     #
    # img_show(if_train=True, a=1)

    # img_show_tuning(if_train=True, n=1, class_num=9)
    # img_show_before_tuning(if_train=True, n=1, class_num=9)

    #
    # # validate code
    from Visualization import class_num
    # class_num()

    #
    from frames_processing import polar_remove_set7_load, polar_remove_load
    # polar_remove_set7_load(aug=True)
    # polar_remove_load(aug=False)

    from Visualization import plot_scatter
    # for j in range(10,12):
    #     n = j*100
    #     plot_scatter(if_train=True, n=n, aspect=(2,1,1), i=3)
    #     j+=1

    # plot_scatter(if_train=True, n=1200, aspect=(2, 1, 1), i=3)

    # $$$$$$$$$$$$######dvs lip
    import DVS_Lip

    # DVS_Lip.Collect_Frames()
    # DVS_Lip.hyper_tuner()

    # import Visualization
    # for i in range(5):
    #     Visualization.lip_img_show(True, i)
    #
    #
    # dataset, data_num = DVS_Lip.initial_load(True)
    # path = DVS_Lip.path_list(True)
    # # c = 10-3
    # c = 1
    # from DVS_Lip import index_arr
    # index_array = index_arr()
    # for n in range(5):
    #     frame_n = DVS_Lip.DVS_Lip(dataset, n, index_array, c = c, train = True, save_path=path.present_path())
    #     frame_n.test()

    # events_dict_pos_time, events_dict_neg_time, label = frame_n.Event_List()
    # print(events_dict_pos_time)
    # ###########$$$$$dvs animals
    import DVS_Animals
    import time
    # dataPath = "C:/Users/ASUS/OneDrive - Nanyang Technological University/datasets/DVSAnimals/data/"
    # DVS_Animals.load_check_items()
    # DVS_Animals.slice(dataPath)
    # DVS_Animals.check_length()

    # DVS_Animals.Collect_Frames()

    # DVS_Animals.hyper_tuner(Aug=False)
    DVS_Animals.hyper_tuner(Aug=True)

    # test

    # dataPath = "C:/Users/ASUS/OneDrive - Nanyang Technological University/datasets/DVSAnimals/data/"
    # dataset = DVS_Animals.AnimalsDvsSliced(dataPath)
    # x = DVS_Animals.gen_train_test(dataPath=dataPath, data_num=dataset.__len__())
    # # index_class_list = x.search()
    # x.stack_frames()

    import Visualization

    # list_A = [0, 45, 188, 191, 210, 229, 305, 438, 511, 513, 552, 590, 628, 663, 666, 723, 761, 837, 842, 853, 856, 875, 894, 920, 958]
    # # for i in list_A:
    # #     Visualization.animals_img_show(n=i)
    # for i in range(19,39):
    #     Visualization.animals_img_show(n=i)
    #`     time.sleep(1)

    # fig = Visualization.animal_img()
    # fig.animals_img_show(n=0, active_stack_data=True,
    #                      display_polar=True)

    # for i in range():
    #     fig.animals_img_show(n=i, active_stack_data=True, display_polar=False)
    #     time.sleep(1)
    # fig.plot_scatter(n=13, aspect=(8, 5, 5), part_e_s=True)

    import interval_measurement
    # measurement = interval_measurement.min_interval()
    # measurement.DVS_128_summary()
    # measurement.sl_animals_summary()

    # cal = interval_measurement.cal_config(1)
    # cal.measure_time()
