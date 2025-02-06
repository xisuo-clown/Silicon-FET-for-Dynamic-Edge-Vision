# measurement: minimum interval between two events
class min_interval:

    def __init__(self):
        import event_stream
        train_set_eve, test_set_eve = event_stream.init_event()

        self.data_num = [1176, 288]
        self.event_list = [train_set_eve, test_set_eve]
        self.indexarr = event_stream.index_arr()

    def DVS_128_summary(self):
        from collections import defaultdict
        from tqdm import tqdm
        import event_stream
        class_compare = defaultdict(list)
        for d in range(2):
            print("In DVS 128, set {}".format(d))
            temp_save = []
            for n in tqdm(range(self.data_num[d]), desc="process"):
                dict_pos_time, dict_neg_time, label = (event_stream.polar_save_as_list
                                                       (self.event_list[d], n, self.indexarr))
                class_compare[int(label)].append(self.cal_min_inter(dict_pos_time[0]))
                # print(label, self.cal_min_inter(dict_pos_time[0]))
            for i in range(11):
                temp_save.append(min(class_compare[i]))
                print("Label {}: minimal interval is {}".format(i, min(class_compare[i])))
            print("minimal interval for entire dataset is {}".format(min(temp_save)))

    def sl_animals_summary(self):
        import DVS_Animals
        from tqdm import tqdm
        from collections import defaultdict
        dataPath = "C:/Users/ASUS/OneDrive - Nanyang Technological University/datasets/DVSAnimals/data/"
        dataset = DVS_Animals.AnimalsDvsSliced(dataPath)
        # print("the length is :{}".format(dataset.__len__()))
        temp_save = []
        class_compare = defaultdict(list)
        print("Calculate interval in sl animals dataset:")
        for i in tqdm(range(dataset.__len__()), desc="Event Processing"):
            events_dict_pos_time, events_dict_neg_time, class_index = dataset.Event_List(i)
            class_compare[int(class_index)].append(self.cal_min_inter(events_dict_pos_time))
            # dict_list = [dict_pos_time[0], dict_neg_time[0]]
        for j in range(19):
            temp_save.append(min(class_compare[j]))
            print("Label {}: minimal interval is {}".format(j, min(class_compare[j])))
        print("minimal interval for entire dataset is {}".format(min(temp_save)))

    def cal_min_inter(self, dict_list):
        temp_list = []

        for i in range(128 * 128):
            try:
                dict_list[i].pop()
                if dict_list[i]:
                    # a = min(dict_list[i])/c
                    temp_list.append(min(dict_list[i]))
                    if min(dict_list[i]) < 1e-4:
                        print(min(dict_list[i]))
            except:
                pass

        min_inter = min(temp_list)
        return min_inter


class cal_config:
    def __init__(self, pw):
        self.pw = pw
        # unit of pw: ms
        self.top_base_ratio = 2

    def cal_frequency(self):
        t = self.pw * 2
        f = 1 / t
        return f

    # def measure_time(self, dc):
    def measure_time(self):
        print("When pulse width equals to {} ms".format(self.pw))
        for i in range(4):
            dc = 0.2 * (i + 1)
            cycle = self.pw * 1E-3 / dc
            f = 1 / cycle
            print("Duty Cycle: {0}, Cycle: {1}, Frequency: {2}".format(dc, cycle, f))
