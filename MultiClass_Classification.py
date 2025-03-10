import numpy as np
import matplotlib.pyplot as plt

N = 100  # 每一个类别的生成的点的数量
D = 2  # 每个点的维度，这里使用平面，所以是2维数据
K = 3  # 类别数量，我们一共生成3个类别的点

# 所有的样本数据，一共300个点，每个点用2个维度表示
# 所有训练数据就是一个300*2的二维矩阵
X = np.zeros((N * K, D))
# 标签数据，一共是300个点，每个点对应一个类别，
# 所以标签是一个300*1的矩阵
y = np.zeros(N * K, dtype='uint8')

# 生成训练数据
for j in range(K):
    ix = range(N * j, N * (j + 1))
    r = np.linspace(0.0, 1, N)
    t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.show()

print("X:", X)
print("Y:", y)

h = 100  # 隐藏层的神经元数量

# 第一个层的权重和偏置初始化
W1 = 0.01 * np.random.randn(D, h)
b1 = np.zeros((1, h))

# 第二层的权重和偏置初始化
W2 = 0.01 * np.random.randn(h, K)
b2 = np.zeros((1, K))

step_size = 1e-0
reg = 1e-3  # regularization strength

# 获取训练样本数量
num_examples = X.shape[0]

for i in range(10000):

    # 计算第一个隐藏层的输出，使用ReLU激活函数
    hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
    # 计算输出层的结果，也就是最终的分类得分
    scores = np.dot(hidden_layer, W2) + b2

    # softmax
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]

    # 计算损失，和之前的一样
    correct_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(correct_logprobs) / num_examples
    reg_loss = 0.5 * reg * np.sum(W1 * W1) + 0.5 * reg * np.sum(W2 * W2)
    loss = data_loss + reg_loss

    if i % 1000 == 0:
        print("iteration %4d loss %f" % (i, loss))

    # 计算scores的梯度
    dscores = probs
    dscores[range(num_examples), y] -= 1
    dscores /= num_examples

    # 计算梯度，反向传播
    dW2 = np.dot(hidden_layer.T, dscores)
    db2 = np.sum(dscores, axis=0, keepdims=True)

    # 反向传播隐藏层
    dhidden = np.dot(dscores, W2.T)
    # 反向传播ReLu函数
    dhidden[hidden_layer <= 0] = 0

    dW1 = np.dot(X.T, dhidden)
    db1 = np.sum(dhidden, axis=0, keepdims=True)

    # 加上正则项
    dW2 += reg * W2
    dW1 += reg * W1

    # 更新参数
    W1 += -step_size * dW1
    b1 += -step_size * db1
    W2 += -step_size * dW2
    b2 += -step_size * db2

# 训练结束，估算正确率
hidden_layer = np.maximum(0, np.dot(X, W1) + b1)
scores = np.dot(hidden_layer, W2) + b2
predicted_class = np.argmax(scores, axis=1)
print("Training accuracy: %.2f" % (np.mean(predicted_class == y)))
