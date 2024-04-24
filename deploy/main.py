import sys
import joblib
import onnxruntime

import numpy as np


def main(args):
    onnx_model_path = "model.onnx"  # onnx模型路径
    # 载入模型加载器
    session = onnxruntime.InferenceSession(onnx_model_path)
    # 载入训练集归一化模型
    x_scaler = joblib.load('x.scaler')
    y_scaler = joblib.load('y.scaler')

    args_inputs = np.array(args).astype(np.float32)  # 格式转换
    session_inputs = {session.get_inputs()[0].name: [args_inputs]}  # 模型输入
    results = session.run(None, session_inputs)  # 模型推理

    print(results[0])


if __name__ == '__main__':
    main(sys.argv[1:])
