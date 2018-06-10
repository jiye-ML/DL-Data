import os
import shutil
import zipfile
import numpy as np
from glob import glob
import scipy.misc as misc

from Tools import Tools


class PreData:

    def __init__(self, zip_file, ratio=4):
        data_path = zip_file.split(".zip")[0]
        self.train_path = os.path.join(data_path, "train")
        self.test_path = os.path.join(data_path, "test")

        if not os.path.exists(data_path):
            f = zipfile.ZipFile(zip_file, "r")
            f.extractall(data_path)

            all_image = self.get_all_images(os.path.join(data_path, data_path.split("/")[-1]))
            self.get_data_result(all_image, ratio, Tools.new_dir(self.train_path), Tools.new_dir(self.test_path))
        else:
            Tools.print_info("data is exists")
        pass

    # 生成测试集和训练集
    @staticmethod
    def get_data_result(all_image, ratio, train_path, test_path):
        train_list = []
        test_list = []

        # 遍历
        Tools.print_info("bian")
        for now_type in range(len(all_image)):
            now_images = all_image[now_type]
            for now_image in now_images:
                # 划分
                if np.random.randint(0, ratio) == 0:  # 测试数据
                    test_list.append((now_type, now_image))
                else:
                    train_list.append((now_type, now_image))
            pass

        # 打乱
        Tools.print_info("shuffle")
        np.random.shuffle(train_list)
        np.random.shuffle(test_list)

        # 提取训练图片和标签
        Tools.print_info("train")
        for index in range(len(train_list)):
            now_type, image = train_list[index]
            shutil.copyfile(image, os.path.join(train_path,
                                                str(np.random.randint(0, 1000000)) + "-" + str(now_type) + ".jpg"))

        # 提取测试图片和标签
        Tools.print_info("test")
        for index in range(len(test_list)):
            now_type, image = test_list[index]
            shutil.copyfile(image, os.path.join(test_path,
                                                str(np.random.randint(0, 1000000)) + "-" + str(now_type) + ".jpg"))

        pass

    # 所有的图片
    @staticmethod
    def get_all_images(images_path):
        all_image = []
        all_path = os.listdir(images_path)
        for one_type_path in all_path:
            now_path = os.path.join(images_path, one_type_path)
            if os.path.isdir(now_path):
                now_images = glob(os.path.join(now_path, '*.jpg'))
                all_image.append(now_images)
            pass
        return all_image

    # 生成数据
    @staticmethod
    def main(zip_file):
        pre_data = PreData(zip_file)
        return pre_data.train_path, pre_data.test_path

    pass


class Data:
    def __init__(self, batch_size, type_number, image_size, image_channel, train_path, test_path):
        self.batch_size = batch_size

        self.type_number = type_number
        self.image_size = image_size
        self.image_channel = image_channel

        self._train_images = glob(os.path.join(train_path, "*.jpg"))
        self._test_images = glob(os.path.join(test_path, "*.jpg"))

        self.test_batch_number = len(self._test_images) // self.batch_size
        pass

    def next_train(self):
        begin = np.random.randint(0, len(self._train_images) - self.batch_size)
        return self.norm_image_label(self._train_images[begin: begin + self.batch_size])

    def next_test(self, batch_count):
        begin = self.batch_size * (0 if batch_count >= self.test_batch_number else batch_count)
        return self.norm_image_label(self._test_images[begin: begin + self.batch_size])

    @staticmethod
    def norm_image_label(images_list):
        images = [np.array(misc.imread(image_path).astype(np.float)) / 255.0 for image_path in images_list]
        labels = [int(image_path.split("-")[1].split(".")[0]) for image_path in images_list]
        return images, labels

    pass

