from NeuralNetwork.neural_network import NeuralNetwork
from NeuralNetwork.layer import Layer
import matplotlib.pyplot as plt
import seaborn as sns #要注意的是一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

N_SAMPLES = 2000 # 采样点数
TEST_SIZE = 0.3 # 测试数量比率
# 利用工具函数直接生成数据集
X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
# 将2000 个点按着7:3 分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=TEST_SIZE, random_state=42)
print(X.shape, y.shape)
# 绘制数据集的分布，X 为2D 坐标，y 为数据点的标签
def make_plot(X, y, plot_name, file_name=None, XX=None, YY=None, preds=None,dark=False):
    if (dark):
        plt.style.use('dark_background')
    else:
        sns.set_style("whitegrid")
    plt.figure(figsize=(16,12))
    axes = plt.gca()
    axes.set(xlabel="$x_1$", ylabel="$x_2$")
    plt.title(plot_name, fontsize=30)
    plt.subplots_adjust(left=0.20)
    plt.subplots_adjust(right=0.80)
    if(XX is not None and YY is not None and preds is not None):
        plt.contourf(XX, YY, preds.reshape(XX.shape), 25, alpha = 1,cmap=plt.cm.Spectral)
        plt.contour(XX, YY, preds.reshape(XX.shape), levels=[.5],cmap="Greys", vmin=0, vmax=.6)
    # 绘制散点图，根据标签区分颜色
    plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), s=40, cmap=plt.cm.Spectral,edgecolors='none')
    # plt.savefig('dataset.svg')
    plt.show()


if __name__ == '__main__':
    # 调用make_plot 函数绘制数据的分布，其中X 为2D 坐标，y 为标签
    make_plot(X, y, "Classification Dataset Visualization ")
    # plt.show()

    nn = NeuralNetwork()  # 实例化网络类
    nn.add_layer(Layer(2, 25, 'tanh'))  # 隐藏层1, 2=>25
    nn.add_layer(Layer(25, 25, 'tanh'))  # 隐藏层1, 25=>25
    nn.add_layer(Layer(25, 25, 'tanh'))  # 隐藏层1, 25=>25
    nn.add_layer(Layer(25, 2, 'sigmoid'))  # 隐藏层2, 25=>2
    learning_rate = 0.01
    max_epochs = 1000
    nn.train(X_train, X_test, y_train, y_test, learning_rate, max_epochs)