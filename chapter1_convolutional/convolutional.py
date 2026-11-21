import numpy as np
import tensorflow as tensorflow

# 使用TensorFlow 1.x的计算图方式
tf = tensorflow.compat.v1
tf.disable_v2_behavior()

# 加载MNIST数据
(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()

# 图像展平并归一化
x_train = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 784).astype(np.float32) / 255.0

# 标签转换成One-Hot
y_train = np.eye(10, dtype=np.float32)[y_train]
y_test = np.eye(10, dtype=np.float32)[y_test]


# 模拟原教程中的next_batch
class DataSet:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def next_batch(self, size):
        indices = np.random.choice(
            len(self.images), size, replace=False
        )
        return self.images[indices], self.labels[indices]


mnist_train = DataSet(x_train, y_train)

"""
以上代码是AI撰写为了版本适应，这个tensor版本太老了
下面的主框架是自己撰写的
"""

#权重初始化
def weight(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)#截断正则化
    return tf.Variable(initial)
#偏置初始化
def bias(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#卷积层
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')#要素x，w，步长，填充
#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,[1,2,2,1],strides=[1,2,2,1],padding='SAME')


if __name__=='__main__':
    #读入数据,已经载入
    # 处理输入数据，需要提供占位，创建空间
    x=tf.placeholder(tf.float32,[None,784])
    y_label=tf.placeholder(tf.float32,[None,10])
    #转换数据维度还原为图片
    x_image=tf.reshape(x,[-1,28,28,1])

    #第一层网络
    w_layer1=weight([5,5,1,32])
    b_layer1=bias([32])
    conv2d_layer1=tf.nn.relu(conv2d(x_image,w_layer1)+b_layer1)#错误1应该输入图像
    pool_2x2_layer1=max_pool_2x2(conv2d_layer1)

    #第二层网络
    w_layer2 = weight([5, 5, 32, 64])
    b_layer2 = bias([64])
    conv2d_layer2 = tf.nn.relu(conv2d(pool_2x2_layer1, w_layer2) + b_layer2)
    pool_2x2_layer2 = max_pool_2x2(conv2d_layer2)

    #全连接层1
    w_fc1=weight([7*7*64,1024])
    b_fc1=bias([1024])
    x_flat=tf.reshape(pool_2x2_layer2,[-1,7*7*64])
    h_fc1=tf.nn.relu(tf.matmul(x_flat,w_fc1)+b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)#忘记dropout了

    #全连接层2
    w_fc2=weight([1024,10])
    b_fc2=bias([10])
    h_fc2=tf.matmul(h_fc1_drop,w_fc2)+b_fc2#是否需要激活函数


    #计算误差
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_label, logits=h_fc2))
    # 同样定义训练模式
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    #准备准确率的计算-标签对比，然后转换为浮点数
    correct_prediction=tf.equal(tf.argmax(y_label,1),tf.argmax(h_fc2,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    sess=tf.InteractiveSession()#创建一个计算图
    sess.run(tf.global_variables_initializer())#对于全局的变量进行初始化

    #训练
    for i in range(2000):
        batch=mnist_train.next_batch(50)#取一批数据
        if i%100==0:#每100组就打印准确率
            train_accuracy=accuracy.eval(feed_dict={x:batch[0],y_label:batch[1],keep_prob:1.0})
            print("step %d,training accuracy %g"%(i,train_accuracy))
        train_step.run(feed_dict={x:batch[0],y_label:batch[1],keep_prob:0.5})

    #测试
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x: x_test, y_label: y_test, keep_prob: 1.0}))






