import os.path
import PIL.Image
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
from skimage.transform import warp_polar
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob
import pandas as pd
import json
from pylab import *
import pickle
import h5py
from mpl_toolkits.basemap import Basemap

class MyDataSetTCIR(Dataset):
    def __init__(self, dataset_dir: str, multi_modal=False) -> None:

        self.dataset_dir = Path(dataset_dir)
        self.multi_modal = multi_modal
        # load "info" as pandas dataframe
        self.data_info = pd.read_hdf(self.dataset_dir, key="info", mode='r')
        self.data_matrix = h5py.File(self.dataset_dir, 'r')['matrix']

        showPosition = 0
        if showPosition == 1:
            all_lon = []
            all_lat = []
            all_vmax = []
            all_time = []

            # Show all the location of the Typhoon
            for i in range(7569):
                vmax = self.data_info.iloc[i].loc['Vmax']
                Lon = self.data_info.iloc[i].loc['lon']
                Lat = self.data_info.iloc[i].loc['lat']
                Time = self.data_info.iloc[i].loc['time']

                all_vmax.append(vmax)
                all_lon.append(Lon)
                all_lat.append(Lat)
                all_time.append(int(Time[:4]))

            all_unique_time = np.unique(all_time)

            # 创建一个地图用于绘制。我们使用的是墨卡托投影，并显示整个世界。
            #m = Basemap(projection='merc', llcrnrlat=-50, urcrnrlat=65, llcrnrlon=-165, urcrnrlon=155, lat_ts=20,
            #            resolution='c')

            m = Basemap(projection='robin', lat_0=0, lon_0=0, resolution='i',
                            area_thresh=5000.0)

            # 绘制海岸线，以及地图的边缘
            m.drawcoastlines()
            m.drawmapboundary()
            m.drawcountries()
            m.drawstates()
            m.drawcounties()

            # Convert coords to projected place in figur
            x, y = m(all_lon, all_lat)
            m.scatter(x, y, 1, marker='.', c=all_vmax, cmap=plt.cm.Set1)
            cb = m.colorbar()

            plt.show()

    def __len__(self):
        return len(self.data_info)

    def AvoidDamagedVal(self, matrix):
        NanVal = np.where(matrix==np.NaN)
        LargeVal = np.where(matrix>1000)
        DemagedVal = [NanVal, LargeVal]
        for item in DemagedVal:
            for idx in range(len(item[0])):
                i = item[0][idx]
                j = item[1][idx]
                allValidPixel = []
                for u in range(-2,2):
                    for v in range(-2,2):
                        if (i+u) < 201 and (j+v) < 201 and not np.isnan(matrix[i+u,j+v]) and not matrix[i+u,j+v] > 1000:
                            allValidPixel.append(matrix[i+u,j+v])
                if len(allValidPixel) != 0:
                    matrix[i][j] = np.mean(allValidPixel)

        return matrix

    def __getitem__(self, index):        
        id = self.data_info.iloc[index].loc['ID']
        vmax = self.data_info.iloc[index].loc['Vmax']
        Lon = self.data_info.iloc[index].loc['lon']
        Lat = self.data_info.iloc[index].loc['lat']
        Time = self.data_info.iloc[index].loc['time']

        # Slice1: IR
        # Slice2: Water vapor
        # Slice3: VIS
        # Slice4: PMW

        ch_slice = self.data_matrix[index][:, :, 0]
        ch_slice1 = self.data_matrix[index][:, :, 1]
        # ch_slice2 = self.data_matrix[index][:, :, 2]
        ch_slice3 = self.data_matrix[index][:, :, 3]

        img = np.zeros((201, 201, 3))
        if self.multi_modal:
            ch_slice = self.AvoidDamagedVal(ch_slice)
            ch_slice1 = self.AvoidDamagedVal(ch_slice1)
            #ch_slice2 = self.AvoidDamagedVal(ch_slice2)
            ch_slice3 = self.AvoidDamagedVal(ch_slice3)

            # img[:, :, 0] = ch_slice # IR
            # img[:, :, 1] = ch_slice1# Water vapor
            # img[:, :, 2] = ch_slice3# PMW
            img[:, :, 0] = ch_slice # IR
            img[:, :, 1] = ch_slice# Water vapor
            img[:, :, 2] = ch_slice3# PMW
        else: 
            img[:, :, 0] = ch_slice
            img[:, :, 1] = ch_slice
            img[:, :, 2] = ch_slice

        img = img.astype(np.uint8)
        img = PIL.Image.fromarray(img)
        return img, vmax, Lon, Lat, Time, id


class MySubset_WS(Dataset):
    """ 划分数据集之后重新设置预处理操作 """

    def __init__(self, dataset: Dataset, transform=None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def _is_leap_year(self, year):
        if year % 4 != 0 or (year % 100 == 0 and year % 400 != 0):
            return False
        return True

    def get_scaled_date_ratio(self, year, month, day):
        r'''
        scale date to [-1,1]
        '''
        days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        total_days = 365
        if self._is_leap_year(year):
            days[1] += 1
            total_days += 1

        assert day <= days[month - 1]
        sum_days = sum(days[:month - 1]) + day
        assert sum_days > 0 and sum_days <= total_days

        # Transform to [-1,1]
        return (sum_days / total_days) * 2 - 1

    def GetItemMultiScale(self, Lon, lat):


        return 1

    def __getitem__(self, index):
        img, label, Lon, Lat, Time, id = self.dataset[index]

        if self.transform is not None:
            img = self.transform(img)

        label = int(label)

        classification_label = Transform_WindSpeed_Classes2(label)

        return img, label, classification_label, Time, id


class MyDataSetDeepTI(Dataset):
    """自定义数据集"""

    def __init__(self, Flag, transform=None):
        # Read From TxTFile
        train_source = 'nasa_tropical_storm_competition_train_source'
        train_labels = 'nasa_tropical_storm_competition_train_labels'

        download_dir = Path("/home/chenhuanxin/datasets/NASA-tropical-storm")

        self.images_path = []
        self.images_class = []

        self.TrainImageList = []
        self.TestImageList = []

        self.TrainImageLabelList = []
        self.TestImageLabelList = []

        if os.path.isfile("AllData.txt"):
            with open('AllData.txt', 'rb') as file2:
                self.TrainImageList = pickle.load(file2)
                self.TrainImageLabelList = pickle.load(file2)
                self.TestImageList = pickle.load(file2)
                self.TestImageLabelList = pickle.load(file2)

        else:
            jpg_names = glob(str(download_dir / train_source / '**' / '*.jpg'))

            for jpg_path in jpg_names:
                self.TrainImageList.append(jpg_path)
                jpg_path = Path(jpg_path)

                # Get the IDs and file paths
                features_path = jpg_path.parent / 'features.json'
                image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])
                storm_id = image_id.split('_')[0]
                labels_path = str(jpg_path.parent / 'labels.json').replace(train_source, train_labels)

                # Load the features data
                with open(features_path) as src:
                    features_data = json.load(src)

                # Load the labels data
                with open(labels_path) as src:
                    labels_data = json.load(src)

                self.TrainImageLabelList.append(int(labels_data['wind_speed']))

            self.images_path = self.TrainImageList
            self.images_class = self.TrainImageLabelList

            test_source = 'nasa_tropical_storm_competition_test_source'
            test_labels = 'nasa_tropical_storm_competition_test_labels'

            jpg_names = glob(str(download_dir / test_source / '**' / '*.jpg'))

            for jpg_path in jpg_names:
                self.TestImageList.append(jpg_path)
                jpg_path = Path(jpg_path)
                # Get the IDs and file paths
                features_path = jpg_path.parent / 'features.json'
                image_id = '_'.join(jpg_path.parent.stem.rsplit('_', 3)[-2:])

                labels_path = str(jpg_path.parent / 'labels.json').replace(test_source, test_labels)

                with open(labels_path) as src:
                    labels_data = json.load(src)

                self.TestImageLabelList.append(int(labels_data['wind_speed']))

            self.images_path = self.TestImageList
            self.images_class = self.TestImageLabelList

            with open('AllData.txt', 'wb') as file:
                pickle.dump(self.TrainImageList, file)
                pickle.dump(self.TrainImageLabelList, file)
                pickle.dump(self.TestImageList, file)
                pickle.dump(self.TestImageLabelList, file)

        self.transform = transform
        if Flag == "Train":
            self.images_path = self.TrainImageList
            self.images_class = self.TrainImageLabelList

        if Flag == "Val":
            self.images_path = self.TestImageList
            self.images_class = self.TestImageLabelList

        if Flag == "Test":
            self.images_path = self.TestImageList
            self.images_class = self.TestImageLabelList

    def __len__(self):
        return len(self.images_path)

    def Usual_GetItem(self, item):
        img = cv2.imread(self.images_path[item])
        img = PIL.Image.fromarray(img)

        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label, 1, 2, 3

    def FANet_GetItem(self, item):
        img = Image.open(self.images_path[item])

        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        # Perform Polar Transform: [3, 224, 224]
        Polar_Image = np.asarray(img)
        # [3, 224, 224] ---> [224, 224, 3]
        Polar_Image = np.transpose(Polar_Image, [1, 2, 0])
        # Polar Transform:
        Polar_Image = warp_polar(Polar_Image, radius=112, multichannel=True)

        Polar_Image = cv2.resize(Polar_Image, (224, 224))
        # [224, 224, 3] ---> [3, 224, 224]
        Polar_Image = np.transpose(Polar_Image, [2, 1, 0])
        Polar_Image = torch.from_numpy(Polar_Image)

        All_Image = torch.cat((img, Polar_Image), 0)

        return All_Image, label

    def __getitem__(self, item):
        return self.Usual_GetItem(item)

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels, dtype=torch.float)

        return images, labels

def Transform_WindSpeed_Classes(WindSpeed):
    # Transform into 7 classes
    # Tropical Depression  <= 33
    # Tropical Storm       34–64
    # Category 1           65–96
    # Category 2           97–110
    # Category 3           111–130
    # Category 4           131–155
    # Category 5           > 155

    if WindSpeed <= 33:
       WSclass = 1
       return WSclass
    if WindSpeed <= 64:
       WSclass = 2
       return WSclass
    if WindSpeed <= 96:
       WSclass = 3
       return WSclass
    if WindSpeed <= 110:
       WSclass = 4
       return WSclass
    if WindSpeed <= 130:
       WSclass = 5
       return WSclass
    if WindSpeed <= 155:
       WSclass = 6
       return WSclass
    else:
       WSclass = 7
       return WSclass

def Transform_WindSpeed_Classes2(WindSpeed):
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
       return WSclass
    if WindSpeed <= 63:
       WSclass = 2
       return WSclass
    if WindSpeed <= 82:
       WSclass = 3
       return WSclass
    if WindSpeed <= 96:
       WSclass = 4
       return WSclass
    if WindSpeed <= 112:
       WSclass = 5
       return WSclass
    if WindSpeed <= 136:
       WSclass = 6
       return WSclass
    else:
       WSclass = 7
       return WSclass

