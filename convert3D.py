import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mlxtend.data import loadlocal_mnist
import pandas as pd

images, labels = loadlocal_mnist(
        images_path='./dataset/train-images.idx3-ubyte',
        labels_path='./dataset/train-labels.idx1-ubyte')

# convert 2D data to 3D data then save to local file
rootDir = "./dataset/3d"
csvList = []

for idx in range(6000):
# for idx in range(1):
    thisImage = images[idx]
    thisLabel = labels[idx]
    thisImage = thisImage.reshape((28, 28))
    thisImagePad = np.zeros((64, 64))
    thisImagePad[18:46, 18:46] = thisImage
    thisImage3D = np.zeros((64, 64, 64))
    for i in range(64):
        thisImage3D[:, :, i] = thisImagePad
    tmpStr = "/" + str(idx) + ".npy"
    print(tmpStr)

    tmpList = []
    tmpList.append(thisLabel)
    tmpList.append(tmpStr)
    csvList.append(tmpList)

    np.save(rootDir+tmpStr, thisImage3D)

    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.voxels(thisImage3D, edgecolor='k')
    # plt.show()

pd.DataFrame.from_dict(csvList).to_csv('./dataset/3d/train.csv', header=False, index=False)