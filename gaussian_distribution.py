import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


# 提取所有 .txt 文件中的 accuracy 值
def extract_accuracy_values(folder_path,tune:bool):
    accuracy_values = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
           if tune and "tune" in file_name or not tune and "tune" not in file_name:
                with open(os.path.join(folder_path, file_name), 'r') as file:
                    content = file.read()
                    matches = re.findall(r'accuracy:\s*([\d.]+)', content)
                    accuracy_values.extend(map(float, matches))
    return accuracy_values


# 绘制高斯分布
def plot_gaussian_distribution(data):
    # 计算均值和标准差
    mean = np.mean(data)
    std_dev = np.std(data)

    # 绘制直方图
    plt.hist(data, bins=20, density=True, alpha=0.6, color='g', label='Data')

    # 生成高斯分布曲线
    x = np.linspace(min(data), max(data), 1000)
    pdf = norm.pdf(x, mean, std_dev)
    plt.plot(x, pdf, 'r', label='Gaussian Fit')

    # 添加图例和标签
    plt.title('Gaussian Distribution of Accuracy')
    plt.xlabel('Accuracy')
    plt.ylabel('Density')
    plt.legend()

    # 显示图像
    plt.show()


# 封装为主程序函数
def analyze_accuracy_distribution(folder_path,tune:bool):
    """
    提取指定文件夹中所有 .txt 文件的 accuracy 值，并绘制高斯分布图。

    参数:
    folder_path (str): 包含 .txt 文件的文件夹路径。
    """
    accuracy_values = extract_accuracy_values(folder_path,tune)

    if accuracy_values:
        print(f"提取到的 Accuracy 值: {accuracy_values}")
        plot_gaussian_distribution(accuracy_values)
    else:
        print("未找到任何 Accuracy 数据。")
