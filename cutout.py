import numpy as np
import cv2


class Cutout(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (H, W, C).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h, w, _ = img.shape

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            # 保证在图片内部
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.
        # 掩码
        img = [img[:, :, i] * mask for i in range(len(img[0, 0, :]))]
        return np.stack(img, 2)


if __name__ == '__main__':

    image = cv2.imread("../images/1.JPEG")
    image = Cutout(1, 300)(image)
    cv2.imwrite("../images/1_.JPEG", image)

