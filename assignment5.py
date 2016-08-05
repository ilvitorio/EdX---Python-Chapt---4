import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
import os

# Look pretty...
matplotlib.style.use('ggplot')

run=1

#
# TODO: Start by creating a regular old, plain, "vanilla"
# python list. You can call it 'samples'.
#
samples = []
color_sample =[]

#
# TODO: Write a for-loop that iterates over the images in the
# Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a
# temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you
# have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no
# effect on the algorithm's results.


path = 'C:\\Users\\U447354\\Documents\\Python Scripts\\Chapter 4\\Isomap\\Datasets\\ALOI\\32'
files = os.listdir(path)

for i in files:
    # Load the image up
    img = misc.imread(path +'\\'+i)
    # Is the image too big? Shrink it by an order of magnitude
    #img = img[::2, ::2]
    samples.append(img.reshape(-1))
    color_sample.append('b')

   


#
# TODO: Once you're done answering the first three questions,
# right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images
# in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 
if run==1:
    path2 = 'C:\\Users\\U447354\\Documents\\Python Scripts\\Chapter 4\\Isomap\\Datasets\\ALOI\\32i'
    files2 = os.listdir(path2)
    for j in files2:
        if j.endswith(".png"):
            # Load the image up
            img = misc.imread(path2 +'\\'+j)
            # Is the image too big? Shrink it by an order of magnitude
            #img = img[::2, ::2]
            samples.append(img.reshape(-1))
            color_sample.append('r')
#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 
df_images = pd.DataFrame(samples)
#df_images_t = df_images.transpose()

#
# TODO: Implement Isomap here. Reduce the dataframe df down
# to three components, using K=6 for your neighborhood size
#
# .. your code here .. 
iso_bear=Isomap(n_components=3,n_neighbors=6)
iso_bear.fit(df_images)
T_iso_bear = iso_bear.transform(df_images)

#
# TODO: Create a 2D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('Manifold Scatterplot')
ax.set_xlabel('Component: {0}'.format(0))
ax.set_ylabel('Component: {0}'.format(1))
ax.scatter(T_iso_bear[:,0],T_iso_bear[:,1], marker='.',alpha=0.7, c=color_sample)



#
# TODO: Create a 3D Scatter plot to graph your manifold. You
# can use either 'o' or '.' as your marker:
#
# .. your code here .. 
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Manifold Component 0')
ax.set_ylabel('Manifold Component 1')
ax.set_zlabel('Manifold Component 2')
ax.scatter(T_iso_bear[:,0], T_iso_bear[:,1], T_iso_bear[:,2], c=color_sample, marker='.')


plt.show()

