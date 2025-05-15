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

def plot_table(data,title):
    import numpy as np
    import matplotlib.pyplot as plt

    # 示例数据
    arr = np.array(data)
    arr = np.insert(arr, 0, title)
    # 创建画布
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis("tight")
    ax.axis("off")

    # 创建表格
    table_data = [[val] for val in arr]  # 变成二维列表，每个元素都是单独的一行
    table = ax.table(cellText=table_data, colLabels=["Values"], loc="center")

    plt.show()


# 绘制高斯分布
def plot_distribution(data,label,title):
    color_table=['r','g','b','c','m','y','k']
    i=0
    for d in data:
        mean = np.mean(d)
        std_dev = np.std(d)
        import seaborn as sns
        # 绘制直方图
        sns.kdeplot(d, fill=True, alpha=0.5, label=label[i], color=color_table[i])
        plt.axvline(mean, color=color_table[i], linestyle='dashed', linewidth=2, label=f'Mean: {mean:.3f}\nVariance: {std_dev:.3f}')
        i+=1




        # 添加图例和标签

    plt.title('Distribution Curve of {}'.format(title))
    plt.xlabel(title)
    plt.ylabel('Density')
    plt.legend()
    plt.show()

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
