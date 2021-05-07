import numpy as np
import scipy.stats

if __name__ == '__main__':
   x = np.array([[1, 2, 3], [1, 4, 5], [1, 3, 3]])
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


   p = np.asarray([0.65, 0.25, 0.07, 0.03])
   q = np.array([0.6, 0.25, 0.1, 0.05])
   q2 = np.array([0.1, 0.2, 0.3, 0.4])


   def JS_divergence(p, q):
      M = (p + q) / 2
      return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


   print(JS_divergence(p, q))  # 0.003093977084273652
   print(JS_divergence(p, q2))  # 0.24719159952098618
   print(JS_divergence(p, p))  # 0.0

