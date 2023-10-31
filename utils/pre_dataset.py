from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import h5py

from torch.utils.data import Dataset

import random
import cv2

def Transform_WindSpeed_Classes2(WindSpeed) -> int:
    # Transform into 7 classes
    # -1 = Tropical depression (W<34)
    # 0 = Tropical storm [34<W<64]
    # 1 = Category 1 [64<=W<83]
    # 2 = Category 2 [83<=W<96]
    # 3 = Category 3 [96<=W<113]
    # 4 = Category 4 [113<=W<137]
    # 5 = Category 5 [W >= 137]

    if WindSpeed <= 33:
       WSclass = 1
    elif WindSpeed <= 63:
       WSclass = 2
    elif WindSpeed <= 82:
       WSclass = 3
    elif WindSpeed <= 96:
       WSclass = 4
    elif WindSpeed <= 112:
       WSclass = 5
    elif WindSpeed <= 136:
       WSclass = 6
    else:
       WSclass = 7
    return WSclass

# 使用h5py的好处：
# 图片不会全部加载到内存当中，节约内存开支，在训练时对数据即取即用，
# 相较于普通的在文件夹中存储图片数据的方式，将图片文件压缩进入h5py
# 文件中，即取即用的效率更高


class TCIRDataSet(Dataset): # 继承pytorch的Dataset类
    def __init__(self, dataset_dir: str) -> None:
        self.dataset_dir = Path(dataset_dir)

        # There are 2 keys in the HDF5:
        # matrix: N x 201 x 201 x 4 HDF5 dataset. One can load this with python numpy.
        # info: HDF5 group. One can load this with python package pandas.
        # 用pandas读取进来，就会是非常容易操作的表格形式


        # load "info" as pandas dataframe
        self.data_info = pd.read_hdf(self.dataset_dir, key="info", mode='r')
        # self.data_info.to_excel('output-CPAC.xls')        

        # load "matrix" as numpy ndarray, this could take longer times
        # with h5py.File(self.dataset_dir, 'r') as hf:
        #     self.data_matrix = hf['matrix'][:]
        # 修改了h5py文件的加载方式，看看能不能加载大文件
        self.data_matrix = h5py.File(self.dataset_dir, 'r')

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        # self.data_info.iloc[index].loc['Vmax'] 可以定位到具体的块
        # info = self.data_info.iloc[index, 2:]
        vmax = self.data_info.iloc[index].loc['Vmax']
        # mslp = self.data_info.iloc[index].loc['MSLP']
        # label = np.array([vmax, mslp])
        # label = Transform_WindSpeed_Classes2(vmax)
        label = vmax
        # img = self.data_matrix['matrix'][index][:,:,0]
        ch_slice = self.data_matrix['matrix'][index][:,:,0]
        img = np.zeros((201, 201, 3))
        img[:,:,0] = ch_slice
        img[:,:,1] = ch_slice
        img[:,:,2] = ch_slice
        img = img.astype(np.uint8)
        return img, label

class MySubset(Dataset):
    """ 划分数据集之后重新设置预处理操作 """

    def __init__(self, dataset: Dataset, transform=None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def main():
    data_path = '/home/chenhuanxin/datasets/TCIR/TCIR-ATLN_EPAC_WPAC.h5'
    ds = TCIRDataSet(data_path)
    print(len(ds))
    # data = ds[0]
    # print(data['info'])
    # print(data['matrix'].shape)

    level = 1
    # tmp = range(0, len(ds))
    for i in range(5):
        img, label = ds[i]
        print(type(img.dtype))
        # print(label)
        # if (Transform_WindSpeed_Classes2(label) == level):
        #     plt.imsave('img-' + str(level) + '-' + str(label) + '.jpg', img)
        #     level += 1
        #     if level == 8:
        #         break
            # cv2.imwrite('img' + str(i) + '.jpg', img)

if __name__ == '__main__':
    main()
