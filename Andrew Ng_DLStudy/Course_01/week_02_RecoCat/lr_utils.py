# Author:司震
# Create: 2024/5/9
# Description: 第X天练习：
import numpy as np
import h5py
def load_dataset():
    train_dataset = h5py.File('F:/ML_practice/Course_01/Andrew Ng_DLStudy/week_02_RecoCat/datasets/train_catvnoncat.h5','r')
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) #获取训练集特征x
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) #获取训练集标签y

    test_dataset = h5py.File('F:/ML_practice/Course_01/Andrew Ng_DLStudy/week_02_Recocat/datasets/test_catvnoncat.h5','r')
    test_set_x_orig = np.array(test_dataset['test_set_x'][:])#获取测试集特征x
    test_set_y_orig = np.array(test_dataset['test_set_y'][:])#获取测试集标签y

    classes =np.array(test_dataset["list_classes"][:]) #保存byte类型的字符串数据['non-cat','cat']

    train_set_y_orig = train_set_y_orig.reshape((1,train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0]))

    return train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes
