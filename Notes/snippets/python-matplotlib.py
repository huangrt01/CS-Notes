%matplotlib widget
import os
import numpy as np
import matplotlib.pyplot as plt

# By default we skip the first row, which contains the headers
# By skipping 2 rows, you can disregard the first data-point (0,0) to get a closer look
def plot_trainings(skiprows=1):
    plt.close()
    for filename in os.listdir("training_data"):
        if filename == ".ipynb_checkpoints": continue
        x, y = np.loadtxt("training_data/" + filename, delimiter=',', unpack=True, skiprows=skiprows)
        plt.plot(x,y, label=filename.split('.csv')[0])

    plt.xlabel('Time (s)')
    plt.ylabel('Validation Accuracy')
    plt.title('Training Comparison')
    plt.legend()
    plt.show()
    
plot_trainings()


# 柱状图

plt.bar(x, y, color='green')

# subplot

plt.subplot(1, 2, 1)
plt.plot(x, y)
plt.title('Left Plot')

plt.subplot(1, 2, 2)
plt.scatter(x, y)
plt.title('Right Plot')

plt.show()