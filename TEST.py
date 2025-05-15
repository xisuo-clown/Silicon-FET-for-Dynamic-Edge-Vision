if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # 定义模型名称、准确率和参数数目
    models = ['mMND-BPTT', 'EGRU', 'AlexNet-LSTM', 'mMND-STDP', 'CNN-LSTM', 'CNN-SNN', 'Vanilla RNN', 'This Work(ResNet)', 'This Work(ResNet-LSTM)']
    accuracies = [98, 97.8, 97.5, 96.6, 93.75, 93.4, 92.01, 94.78, 97.34]
    params = [1.1, 4.8, 8.3, 0.81, 11.4, 2.32, 2.85,0.098,0.197]

    rnn_indices = [1,2,4, 6]  # mMND-BPTT, EGRU, Vanilla RNN
    cnn_indices = [0, 3, 5]  # AlexNet-LSTM, CNN-LSTM, CNN-SNN

    # 创建图像
    plt.figure(figsize=(10, 6))
    x_off=0.35
    y_off=0.1
    # 绘制 RNN 模型的散点，颜色为绿色
    for i in rnn_indices:
        plt.scatter(accuracies[i], params[i], color='green')
        plt.text(accuracies[i]+x_off, params[i]+y_off*params[i], models[i], fontsize=8, ha='right', color='black')

    # 绘制 CNN 模型的散点，颜色为蓝色
    for i in cnn_indices:
        if models[i] == 'mMND-BPTT':
            plt.scatter(accuracies[i], params[i], color='blue')
            plt.text(accuracies[i] + x_off-0.1, params[i] + y_off * params[i], models[i], fontsize=8, ha='right',
                     color='black')
            continue
        plt.scatter(accuracies[i], params[i], color='blue')
        plt.text(accuracies[i]+x_off, params[i]+y_off*params[i], models[i], fontsize=8, ha='right', color='black')

    # 绘制其他模型的散点
    for i in range(len(models)):
        if i not in rnn_indices and i not in cnn_indices:
            plt.scatter(accuracies[i], params[i], color='orange')
            plt.text(accuracies[i]+x_off, params[i]+y_off*params[i], models[i], fontsize=8, ha='right', color='black')

    # 设置坐标轴标签
    plt.ylabel('Parameters (M)')
    plt.xlabel('Accuracy (%)')

    # 设置y轴为对数刻度
    plt.yscale('log')

    # 设置y轴刻度只显示 1 和 10
    plt.yticks([1, 10], ['1', '10'])

    # 设置标题
    plt.title('Model Accuracy vs Parameters')

    # 显示图形
    plt.show()



