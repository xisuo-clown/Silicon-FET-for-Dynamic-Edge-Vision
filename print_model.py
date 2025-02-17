import glob
import os
from collections import Counter
import numpy as np

if __name__ == '__main__':
    print("this is a test")
    # from params_adjustment import find_path
    # root_dir, train_save_path, test_save_path=find_path("tune_for_all")
    # train_x=np.load(os.path.join(train_save_path, "Aug_dataset_features_remove.npy"))
    # train_y=np.load(os.path.join(train_save_path, "Aug_dataset_labels_remove.npy"))
    # counter=Counter(train_y)
    # print(counter)
    # x_for_each_label={i:[] for i in range(10)}
    # [x_for_each_label[train_y[i]].append(train_x[i]) for i in range(len(train_y))]
    # contrast=[]
    # for k,v in x_for_each_label.items():
    #     cst=np.empty(2)
    #     for j in v:
    #         cst+=np.std(j,axis=(0,1))
    #     cst/=len(v)
    #     contrast.append(cst)
    # print(contrast)

    # model_path=os.path.join(test_save_path, "test_results")
    # h5_files = glob.glob(os.path.join(model_path,"*.h5"))
    # from params_adjustment import ResNet18
    # model = ResNet18(10)
    # model.build(input_shape=(None, 128, 128, 2))
    # model.build_graph().summary()