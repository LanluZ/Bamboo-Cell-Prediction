import os
import sys

import numpy as np

from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QApplication

import func.model

# 环境获取
path = os.path.dirname(sys.argv[0])
ui_path = os.path.join(path, 'ui', 'gui.ui')
onnx_model_path = os.path.join(path, 'model', 'model.onnx')
x_scaler_path = os.path.join(path, 'model', 'x_scaler.pkl')
y_scaler_path = os.path.join(path, 'model', 'y_scaler.pkl')

# 实例化UI加载器
loader = QUiLoader()


# 初始化UI类
class GUI:
    def __init__(self):
        self.ui = loader.load(os.path.join(path, 'ui', 'gui.ui'))
        self.ui.calculate_button.clicked.connect(self.calculate_button_clicked)
        # 初始化模型
        self.model = func.model.Model(onnx_model_path, x_scaler_path, y_scaler_path)

    def calculate_button_clicked(self):
        # 字符串解析
        inputs = self.ui.start_data.toPlainText()
        inputs = [float(x) for x in inputs.split(',')]
        inputs = np.array([inputs]).astype(np.float32)

        # 载入模型计算
        result = self.model.predict(inputs)

        # 结果显示
        self.ui.output_data.setText(str(result[0]))


if __name__ == '__main__':
    # 创建应用
    app = QApplication([])
    gui = GUI()
    gui.ui.show()
    app.exec()
