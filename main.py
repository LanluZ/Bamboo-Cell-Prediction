# 导入窗口控制器
from controller import Controller
# 导入布局界面
from ui import Win

if __name__ == '__main__':
    # 实例化一个窗口 将窗口控制器的实例传入
    app = Win(Controller())

    # 在这可对窗口操作 设置图标等.

    # 运行程序
    app.mainloop()
