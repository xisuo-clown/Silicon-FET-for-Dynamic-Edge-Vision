import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



def plot_box(data,title):
    # 设置图形
    plt.figure(figsize=(10, 6))

    classes = len(data)  # 类别数
    samples_per_class = len(data[0])  # 每类样本数

    # 为每一类绘制点状图
    for i in range(classes):
        # 转换当前类的数据为浮点数
        class_data = list(map(float, data[i]))

        # 绘制每个类的散点图
        plt.scatter(np.ones(samples_per_class) * i, class_data, label=f'Class {i + 1}', alpha=0.6, s=50)

        # 计算并显示均值
        class_mean = np.mean(class_data)

        # 设置显示均值的位置，避免与数据点重叠
        plt.text(i, 0.93, f'{class_mean:.2f}', ha='center', va='bottom', fontsize=10, color='black')

    # 添加标签和标题
    plt.title('Scatter Plot for Each Class with Small Sample and Mean for {}'.format(title))
    plt.xlabel('Class')
    plt.ylabel('Value')
    plt.xticks(np.arange(classes), [f'Class {i + 1}' for i in range(classes)])
    plt.legend(title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')

    # 显示图形
    plt.tight_layout()
    plt.show()


def csv_load(suffix:str):
    import numpy as np
    from params_adjustment import find_path
    data=[[] for _ in range(10)]
    root_dir, train_save_path, test_save_path=find_path(suffix)
    results_path=os.path.join(test_save_path, "test_results_no_normalization")
    csv_file=[f for f in os.listdir(results_path) if ".csv" in f]
    for csv in csv_file:
        with open(os.path.join(results_path,csv)) as f:
            df=f.readlines()
        for i in range(len(df)):
            df[i]=df[i].strip().split(',')
        for i in range(10):
            data[i].append(df[i][i])
    return data

def cal_for_brightness(suffix:str):
    x,y=load_npy_data(suffix)
    brightness_mean=np.mean(x,axis=(1,2,3))
    x_mean = [[] for _ in range(10)]
    x_g = classification_for_label(brightness_mean, x, y)
    for i in range(10):
        x_mean[i]=np.mean(x_g[i])
    x_overall=np.mean(np.array(x_g))
    return x_mean,x_overall


def classification_for_label(cal_results, x, y):
    x_g = [[] for _ in range(10)]
    for i in range(len(x)):
        x_g[int(y[i])].append(cal_results[i])
    return x_g


def load_npy_data(suffix):
    from params_adjustment import find_path
    root_dir, train_save_path, test_save_path = find_path(suffix)
    return np.load(os.path.join(train_save_path, "Aug_dataset_features.npy")),np.load(os.path.join(train_save_path, "Aug_dataset_labels.npy"))


def cal_for_contrast(suffix:str):
    x,y=load_npy_data(suffix)
    contrast_mean=np.std(x,axis=(1,2))/np.mean(x,axis=(1,2))
    x_mean=[[] for _ in range(10)]
    x_g=classification_for_label(contrast_mean, x, y)
    for i in range(10):
        x_mean[i]=np.mean(x_g[i])
    x_overall = np.mean(np.array(x_g))
    return x_mean,x_overall


def create_csv(data):
    import csv
    # 示例列表（二维）

    # 将列表写入 CSV 文件
    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print("CSV file has been written.")




if __name__ == '__main__':
    c1=csv_load("no_tune")
    c2=csv_load("tune_for_all")
    # plot_box(csv_load("no_tune"),"without tuning")
    # plot_box(csv_load("tune_for_all"), "with tuning")
    # brightness_mean=[0,0]
    # brightness_overall = [0, 0]
    # contrast_mean = [0, 0]
    # contrast_overall = [0, 0]
    # brightness_mean[0],brightness_overall[0]=cal_for_brightness("no_tune")
    # brightness_mean[1],brightness_overall[1]=cal_for_brightness("tune_for_all")
    # contrast_mean[0],contrast_overall[0]=cal_for_contrast("no_tune")
    # contrast_mean[1],contrast_overall[1]=cal_for_contrast("tune_for_all")
    create_csv(c1)
    create_csv(c2)
    print()





