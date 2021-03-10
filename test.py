import numpy as np

if __name__ == '__main__':
   array = np.array([[1, 2], [1, 3], [1, 4]])
   array = array * array
   weight = np.array([4, 2])
   print(len(array))
   print(weight[1]*array[:, 1] + weight[0])

   y = weight * array
   print(y)
   y_ = np.sum(y, axis=1)
   # y = y[:, 0] + y[:, 1]
   s = 1.0 / (1.0 + np.exp(-100))
   print(s)