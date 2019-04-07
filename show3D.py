import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# file_name = "./mesh/1200.npy"
# # file_name = "./dataset/3d/0.npy"
# image3D = np.load(file_name)
# for idx in range(64):
#     sample = np.zeros((64, 64, 64))
#     for i in range(64):
#         for j in range(64):
#             for k in range(64):
#                 if image3D[idx, 0, i, j, k] > 0.5:
#                     sample[i, j, k] = 255
#                 else:
#                     sample[i, j ,k] = 0
#
#     # sample = sample.transpose(2, 0, 1)
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     ax.voxels(sample[:, :, :], edgecolor='k')
#     print(image3D.shape)
#     print(np.unique(image3D[0, 0, :, :, :]))
#     ax.view_init(elev=70, azim=-30)
#     plt.savefig("./mesh/res/" + str(idx) + ".png")
#     # plt.show()

file_name = "./dataset/chairs/10.npy"

image3D = np.load(file_name)
print(image3D)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.voxels(image3D[:, :, :], edgecolor='k')
ax.view_init(elev=70, azim=-30)
plt.show()