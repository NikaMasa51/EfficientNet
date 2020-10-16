import os, random, cv2
import numpy as np
from albumentations import Compose
from albumentations.augmentations.transforms import HorizontalFlip, Normalize

class DataLoader(object):
    def __init__(self, data_folder, batch_size, input_shape, do_augmentation, gray_scale=False):
        self._file_paths = []
        self._annotations = []
        folders = os.listdir(data_folder)
        folders.sort()
        # 画像のパスとクラスIDを取得する
        for class_id, class_name in enumerate(folders):
            folder_path = data_folder + os.sep + class_name
            file_paths = [folder_path + os.sep + fn for fn in os.listdir(folder_path + os.sep)]
            self._file_paths += file_paths
            self._annotations += [class_id]*len(file_paths)
        # クラス数
        self._class_count = class_id + 1
        self._batch_size = batch_size
        self._input_shape = input_shape
        self._gray_scale = gray_scale
        if len(self._file_paths)%self._batch_size == 0:
            self.iterations = len(self._file_paths) // self._batch_size
            self._has_extra_data = False
        else:
            self.iterations = len(self._file_paths) // self._batch_size + 1
            self._has_extra_data = True
        self._compose = self._define_augment(input_shape, do_augmentation)

    def _define_augment(self, input_shape, do_augmentation):
        # mean, std, max_pixel_valueは適宜変更してください
        mean = (0.485*255, 0.456*255, 0.406*255)
        std = (0.229*255, 0.224*255, 0.225*255)
        normalize = Normalize(mean=mean, std=std, max_pixel_value=1)
        # データ拡張内容は適宜変更してください
        if do_augmentation:
            return Compose([normalize, HorizontalFlip(p=0.5)])
        else:
            return Compose([normalize])

    def get_data_loader(self):
        while True:
            file_paths, annotations = self._shuffle(self._file_paths, self._annotations)
            for iteration in range(self.iterations):
                if iteration == self.iterations - 1 and self._has_extra_data:
                    shape = (len(file_paths)%self._batch_size, self._input_shape[0],
                             self._input_shape[1], self._input_shape[2])
                else:
                    shape = (self._batch_size, self._input_shape[0], self._input_shape[1], self._input_shape[2])
                # バッチサイズ分のデータを取得する
                X = np.zeros(shape, dtype=np.float32)
                y = np.zeros((shape[0], self._class_count), dtype=np.float32)
                for i in range(X.shape[0]):
                    index = self._batch_size*iteration + i
                    if self._gray_scale:
                        image = cv2.imread(file_paths[index], cv2.IMREAD_GRAYSCALE)
                        image = image[:,:,np.newaxis]
                    else:
                        image = cv2.imread(file_paths[index])
                    image = cv2.resize(image, (self._input_shape[1], self._input_shape[0]))
                    image = image.astype(np.float32)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # changesd
                    # データ拡張を実行する
                    X[i] = self._augment(image)
                    y[i, annotations[index]] = 1
                yield X, y

    def _shuffle(self, x, y):
        p = list(zip(x, y))
        random.shuffle(p)
        return zip(*p)

    def _augment(self, image):
        dict = {'image': image}
        augmented = self._compose(**dict)
        return augmented['image']
