import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from skimage import io
from skimage.transform import resize
from skimage.filters import threshold_mean
import os
from sklearn.decomposition import PCA


directory = r'C:\Users\ry\Google Drive\Studia\VII Semestr\MROiUM\MRO\Zad5\imgs'
components = 2
scale = 4
images1D = []
images1D = np.empty((15,129600))
i = 0
img = None
for filename in os.listdir(directory):
    img = io.imread(directory + '\\' + filename, as_grey=True)
    thresh = threshold_mean(img)
    img = img > thresh
    image_resized = resize(img, (img.shape[0] / scale, img.shape[1] / scale))
    flat = np.ndarray.flatten(image_resized)
    images1D[i,:] = flat
    img = flat
    i += 1

pca = PCA(n_components=components)
reduced = pca.fit_transform(np.array(images1D))


print(pca.components_)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.singular_values_ )
print(reduced.shape)
fig = plt.figure(0)
ax = fig.add_subplot(111)

# principal = pca.components_[0]
principal = pca.inverse_transform(reduced[0])
principal = np.reshape(principal, (-1, 480))

ax.imshow(principal, cmap='gray')
# ax.set_title("Resized image (no aliasing)")

plt.figure(1)
plt.scatter(reduced[:,0], reduced[:,1], edgecolor='k', s=20)

plt.show()
