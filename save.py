#Asi se hace


import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

path = "/home/moha/Desktop/CLIP_prefix_caption/data/padchestgr/PadChest_GR/222911790629161189231580845551828037_3lrfjn.png"

# --- load 16-bit PNG correctly ---
img = Image.open(path)
arr = np.array(img)

print(f"array dtype: {arr.dtype}, min: {arr.min()}, max: {arr.max()}")

# if 16-bit, normalize to [0,1]
if arr.dtype == np.uint16:
    arr = arr.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

# visualize
plt.figure(figsize=(6,6))
plt.imshow(arr, cmap="gray")
plt.axis("off")
plt.show()

# optionally convert to 8-bit PIL image
arr8 = (arr * 255).astype(np.uint8)
img8 = Image.fromarray(arr8)
img8.save("padchest_xray_8bit.png")
