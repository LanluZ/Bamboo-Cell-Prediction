from tkinter.messagebox import showinfo

# 导入UI 将 Controller 的属性 ui 类型设置成 Win
from ui import Win

from time_feature import main
from feature_time import main


class Controller:
    # 导入UI类后，替换以下的 object 类型，将获得 IDE 属性提示功能
    ui: Win

    def __init__(self):
        pass

    def button_click_mouse(self, *args):
        print(1)
