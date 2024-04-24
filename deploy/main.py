import pickle
import sys
import onnxruntime

import numpy as np

from sklearn.preprocessing import MinMaxScaler


def main(args):
    onnx_model_path = "model.onnx"  # onnx模型路径
    # 载入模型加载器
    session = onnxruntime.InferenceSession(onnx_model_path)
    # 载入训练集归一化模型
    x_scaler, y_scaler = 0, 0
    with open('x_scaler.pkl', 'rb') as f:
        x_scaler = pickle.load(f)
    with open('y_scaler.pkl', 'rb') as f:
        y_scaler = pickle.load(f)

    args_inputs = np.array(args).astype(np.float32).reshape(1, -1)  # 格式转换
    args_inputs = x_scaler.transform(args_inputs)  # 归一化
    session_inputs = {session.get_inputs()[0].name: args_inputs}  # 模型输入

    results = session.run(None, session_inputs)  # 模型推理

    results = y_scaler.inverse_transform(results[0])  # 反归一化

    print(results[0])


if __name__ == '__main__':
    main(sys.argv[1:])
