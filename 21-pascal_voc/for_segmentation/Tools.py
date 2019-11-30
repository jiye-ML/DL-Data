import time
import os
from tensorflow.python import pywrap_tensorflow
import numpy as np
from matplotlib import pyplot as plt
import cv2


class Tools:
    def __init__(self):
        pass

    @staticmethod
    def print_info(info):
        print(time.strftime("%H:%M:%S", time.localtime()), info)
        pass

    # 新建目录
    @staticmethod
    def new_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @staticmethod
    def print_ckpt(ckpt_path):
        reader = pywrap_tensorflow.NewCheckpointReader(ckpt_path)
        var_to_shape_map = reader.get_variable_to_shape_map()

        for key in var_to_shape_map:
            print("tensor_name: ", key)
            print(reader.get_tensor(key))
            pass
        pass

    pass



