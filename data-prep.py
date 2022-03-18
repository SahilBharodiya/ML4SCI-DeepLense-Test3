# %%
import numpy as np
import os
import cv2


# %%
d = os.listdir('lens_data')


# %%
X = []
y = []


# %%
j = 1
for i in d:
    data = np.load('lens_data/' + i, allow_pickle=True)
    img = data[0]
    img = img / 255.0
    img = cv2.resize(img, (64, 64))
    img = np.stack((img, img, img), axis=-1)
    X.append(img)
    y.append(data[1])
    print(j)
    j += 1


# %%
X = np.array(X)
y = np.array(y)


# %%
np.save('X.npy', X)
np.save('y.npy', y)
