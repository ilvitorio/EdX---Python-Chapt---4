import pandas as pd
import numpy as np
import scipy.io
import random, math

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D

def Plot2D(T, title, x, y, num_to_plot=40):
  # This method picks a bunch of random samples (images in your case)
  # to plot onto the chart:
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title(title)
  ax.set_xlabel('Component: {0}'.format(x))
  ax.set_ylabel('Component: {0}'.format(y))
  x_size = (max(T[:,x]) - min(T[:,x])) * 0.08
  y_size = (max(T[:,y]) - min(T[:,y])) * 0.08
  for i in range(num_to_plot):
    img_num = int(random.random() * num_images)
    x0, y0 = T[img_num,x]-x_size/2., T[img_num,y]-y_size/2.
    x1, y1 = T[img_num,x]+x_size/2., T[img_num,y]+y_size/2.
    img = df.iloc[img_num,:].reshape(num_pixels, num_pixels)
    ax.imshow(img, aspect='auto', cmap=plt.cm.gray, interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))

  # It also plots the full scatter:
  ax.scatter(T[:,x],T[:,y], marker='.',alpha=0.7)



# A .MAT file is a .MATLAB file. The faces dataset could have came
# in through .png images, but we'll show you how to do that in
# anither lab. For now, you'll see how to import .mats:
mat = scipy.io.loadmat('Datasets/face_data.mat')
df = pd.DataFrame(mat['images']).T
num_images, num_pixels = df.shape
num_pixels = int(math.sqrt(num_pixels))

# Rotate the pictures, so we don't have to crane our necks:
for i in range(num_images):
  df.loc[i,:] = df.loc[i,:].reshape(num_pixels, num_pixels).T.reshape(-1)


#
# TODO: Implement PCA here. Reduce the dataframe df down
# to THREE components. Once you've done that, call Plot2D.
#
# The format is: Plot2D(T, title, x, y, num_to_plot=40):
# T is your transformed data, NDArray.
# title is your chart title
# x is the principal component you want displayed on the x-axis, Can be 0 or 1
# y is the principal component you want displayed on the y-axis, Can be 1 or 2
#
pca_data =PCA(n_components=3)
pca_data.fit(df)
T_pca = pca_data.transform(df)
Plot2D(T_pca,'PCA Transformed Data PC0VsPC1',0,1)
#Plot2D(T_pca,'PCA Transformed Data PC0VsPC2',0,2)
#Plot2D(T_pca,'PCA Transformed Data PC1VsPC2',1,2)
#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to THREE components. Once you've done that, call Plot2D using
# the first two components.
#

iso_data = Isomap(n_neighbors=3,n_components=3)
iso_data.fit(df)
T_iso = iso_data.transform(df)
Plot2D(T_iso,'Isomap Transformed Data Ax0VsAx1',0,1)
#Plot2D(T_iso,'Isomap Transformed Data Ax0VsAx2',0,2)
#Plot2D(T_iso,'Isomap Transformed Data Ax1VsAx2',1,2)

#
# TODO: If you're up for a challenge, draw your dataframes in 3D
# Even if you're not, just do it anyway.
#

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.set_xlabel('Principal Component 0')
#ax.set_ylabel('Principal Component 1')
#ax.set_zlabel('Principal Component 2')
#ax.scatter(T_pca[:,0], T_pca[:,1], T_pca[:,2], c='r', marker='.')

#fig1 = plt.figure()
#ax1 = fig1.add_subplot(111, projection='3d')
#ax1.set_xlabel('Manifold Component 0')
#ax1.set_ylabel('Manifold Component 1')
#ax1.set_zlabel('Manifold Component 2')
#ax1.scatter(T_iso[:,0], T_iso[:,1], T_iso[:,2], c='r', marker='.')


plt.show()
