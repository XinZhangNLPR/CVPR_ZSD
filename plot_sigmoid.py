
import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
 
    y=1/(1+np.exp(-x))
    #dy=y*(1-y)
    return y
 
 
def plot_sigmoid():
    # param:起点，终点，间距
    x = np.arange(-8, 8, 0.2)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.show()
 
 
if __name__ == '__main__':
    plot_sigmoid()
