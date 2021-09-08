'''
将voc标签从 [0-255]转化为[0-20]， 虽然voc标签是彩色的，但是是8位深，如果按照三个通道读取
出来的结果，就和这里画板上的一一对应
'''
import os
import sys
from skimage.io import imread, imsave
import numpy as np

# pacal画板对应关系
def pascal_palette():
    palette = {(0, 0, 0): 0,
               (128, 0, 0): 1,
               (0, 128, 0): 2,
               (128, 128, 0): 3,
               (0, 0, 128): 4,
               (128, 0, 128): 5,
               (0, 128, 128): 6,
               (128, 128, 128): 7,
               (64, 0, 0): 8,
               (192, 0, 0): 9,
               (64, 128, 0): 10,
               (192, 128, 0): 11,
               (64, 0, 128): 12,
               (192, 0, 128): 13,
               (64, 128, 128): 14,
               (192, 128, 128): 15,
               (0, 64, 0): 16,
               (128, 64, 0): 17,
               (0, 192, 0): 18,
               (128, 192, 0): 19,
               (0, 64, 128): 20}

    return palette


def convert_from_color_segmentation(arr_3d):
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    palette = pascal_palette()

    for color, label in palette.items():
        temp_arr = np.array(color).reshape(1, 1, 3)
        temp_arr_boolean = np.equal(arr_3d, temp_arr)
        m = np.all(temp_arr_boolean, axis=2)
        arr_2d[m] = label
    return arr_2d


def main():
    # 数据目录
    data_dir = 'E:/data/voc/VOCdevkit/VOC2012'
    # 标签
    path = os.path.join(data_dir, 'SegmentationClass')
    # 转换[0-20]之间的标签
    path_converted = os.path.join(data_dir, "SegmentationClassConverted")
    txt_file = os.path.join(data_dir, "ImageSets/Segmentation/val.txt")

    # Create dir for converted labels
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    with open(txt_file, 'r') as f:
        # 读入文件一行数据
        for img_name in f:
            # 找到标签
            img_base_name = img_name.strip().split()
            # img_base_name = img_base_name[1].split("/")[-1]
            img_base_name = img_base_name[0] + ".png"
            img_name = os.path.join(path, img_base_name)
            # 读取图片，3个通道，这样就可以和画版意义对应
            img = imread(img_name)[:, :, :3]

            if (len(img.shape) > 2):
                # 将][0-255]之间的图片对应到[0-20]
                img = convert_from_color_segmentation(img)
                # 保存
                imsave(os.path.join(path_converted, img_base_name), img)
            else:
                print(img_name + " is not composed of three dimensions, therefore "
                                 "shouldn't be processed by this script.\n Exiting.", file=sys.stderr)
                exit()


if __name__ == '__main__':
    main()
