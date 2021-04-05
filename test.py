import numpy as np

if __name__ == '__main__':
   x = np.array([[1, 0, 0], [0, 0, 1], [0, 0, 1]])
   weight = np.array([4, 2, 3])

   print("x:", x)
   print("w", weight)
   y = x * weight
   print("y = x*w:", y)
   y = np.sum(y, axis=1)
   print("y sum:", y)
   X = range(10)
   print("X", X)
   print(np.random.choice(range(10), 10, replace=False))

   y_label = np.array([0, 2, 1, 2, 0, 0])
   y_onehot = np.zeros((y_label.shape[0], 3))
   y_onehot[np.arange(y_label.shape[0]), y_label] = 1

   print("y_onehot:", y_onehot)

