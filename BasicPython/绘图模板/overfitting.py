import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x1 = np.arange(-10, 10,0.001)
y1 = x1 * np.sin(x1)

x2 = np.array([-7.98, -1.91, 0.04, 4.83])
y2 = np.array([ 7.97, 1.81, 0.06, -4.91])

x3 = np.arange(-10, 10,0.001)
y3 = -x3

TITLE_FONT_SIZE = 12

config = {
    "font.family": 'serif',
    "font.size": 12,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
plt.rcParams.update(config)

plt.subplot(1,2,1)
plt.plot(x1, y1)
plt.scatter(x2,y2, color = "r")
plt.title("假设模型", fontsize = TITLE_FONT_SIZE)
plt.subplot(1,2,2)
plt.plot(x3, y3)
plt.scatter(x2,y2, color = "r")
plt.title("实际模型", fontsize = TITLE_FONT_SIZE)
plt.show()