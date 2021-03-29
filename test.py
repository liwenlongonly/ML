import numpy as np

if __name__ == '__main__':
   x = np.array([[1, 2, 4], [1, 3, 5], [1, 4, 6]])
   weight = np.array([4, 2, 3])

   print("x:", x)
   print("w", weight)
   y = x * weight
   print("y = x*w:", y)
   y = np.sum(y, axis=1)
   print("y sum:", y)
