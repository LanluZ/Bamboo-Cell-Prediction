# 导入UI 将 Controller 的属性 ui 类型设置成 Win
import os
import ui

from ui import Win


class Controller:
    # 导入UI类后，替换以下的 object 类型，将获得 IDE 属性提示功能
    ui: Win

    def __init__(self):
        pass

    def init(self, view):
        self.view = view

    def button_click_mouse(self, *args):
        print(self.view.tk_select_box_1.get()) # 获取选项框
