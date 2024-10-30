import json
import numpy as np
from PIL import Image #for displaying image snippet
import matplotlib.pyplot as plt

f = open(r'./planes dataset/planesnet.json')
planesnet = json.load(f)
f.close()
print(planesnet.keys())

index = 10000 # Row to be saved
im = np.array(planesnet['data'][index]).astype('uint8')
im = im.reshape((3, 400)).T.reshape((20,20,3))
print(im.shape)

plt.imshow(im)
plt.show()

#Save image; may be used when plane is found in segmented image searching
# out_im = Image.fromarray(im)
# out_im.save('test.png')

#Citation for code
# Rhamell. "Displaying and Saving Image Chips." Kaggle, 7 years ago. https://www.kaggle.com/code/rhamell/displaying-and-saving-image-chips