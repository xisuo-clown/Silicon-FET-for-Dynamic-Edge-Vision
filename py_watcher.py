import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from data_analyze import show_pic_by_content
import numpy as np
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "Aug*.npy")])
    output=np.load(file_path)
    for j in range(6):
        for i in range(6):
            show_pic_by_content(output[j,i,:,:,0])


# 创建主窗口
root = tk.Tk()
root.title("图片选择器")

# 按钮
btn = tk.Button(root, text="选择图片", command=select_image)
btn.pack(pady=10)

# 显示图片的 Label
label = tk.Label(root)
label.pack()

# 运行应用
root.mainloop()
