import onnxruntime
import joblib

from sklearn.preprocessing import MinMaxScaler


# onnx模型预测类
class Model:
    def __init__(self, onnx_model_path, x_scaler_path, y_scaler_path):
        # onnxruntime模型加载器
        self.session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])
        # 归一化模型载入
        self.x_scaler = joblib.load(x_scaler_path)
        self.y_scaler = joblib.load(y_scaler_path)

    # 模型预测方法
    def predict(self, inputs):
        # 输入集归一化
        inputs = self.x_scaler.transform(inputs)
        # 模型输入构建
        session_input = {self.session.get_inputs()[0].name: inputs}
        # 模型运行
        session_output = self.session.run(None, session_input)
        # 反归一化
        result = self.y_scaler.inverse_transform(session_output[0])

        return result
