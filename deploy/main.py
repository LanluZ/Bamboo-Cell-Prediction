import sys
import onnxruntime

import numpy as np


def main(args):
    onnx_model_path = "model.onnx"  # onnx模型路径
    session = onnxruntime.InferenceSession(onnx_model_path)  # 载入模型加载器
    args_inputs = np.array(args).astype(np.float32)  # 格式转换
    session_inputs = {session.get_inputs()[0].name: [args_inputs]}  # 模型输入
    results = session.run(None, session_inputs)  # 模型推理

    print(results)


if __name__ == '__main__':
    main(sys.argv[1:])
