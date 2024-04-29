import os

from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication

# 环境获取
path = os.path.dirname(__file__)

# 实例化UI加载器
loader = QUiLoader()


# 初始化UI类
class GUI:
    def __init__(self):
        self.ui = loader.load(os.path.join(path, 'ui', 'gui.ui'))
        self.ui.show()


if __name__ == '__main__':
    # 创建应用
    app = QApplication([])
