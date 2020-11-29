import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    a = np.load("/Users/ashwin/Downloads/noise_amp_db (1).npy")
    print(a.shape)
    b = a[0]
    plt.imshow(b, cmap='hot', interpolation='nearest')
    plt.gca().invert_yaxis()
    plt.show()

