import numpy as np
import matplotlib.pyplot as plt



img_1 = np.load('/media/morzh/ext4_volume/work/StyleFlow/001.npy')
img_2 = np.load('/media/morzh/ext4_volume/work/StyleFlow/002.npy')

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(img_1)
plt.subplot(1, 2, 2)
plt.imshow(img_2)
plt.tight_layout()
plt.show()