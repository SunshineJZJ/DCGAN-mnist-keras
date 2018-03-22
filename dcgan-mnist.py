'''
Name: DCGAN

参数调整：
dropout = 0.2（0.4）


'''

import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

"""时间计数"""
class ElapsedTimer(object):
    def __init__(self):
        self.start_time = time.time()       #开始计时，秒制
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"

    def elapsed_time(self):
        print("Elapsed: %s " % self.elapsed(time.time() - self.start_time) )

class Net(object): 
    def __init__(self, img_rows=28, img_cols=28, channel=1):

        self.img_rows = img_rows
        self.img_cols = img_cols
        self.channel = channel
        self.D = None   # discriminator
        self.G = None   # generator
        self.AM = None  # adversarial model
        self.DM = None  # discriminator model

    # (W−F+2P)/S+1
    """判别器网络"""
    def discriminator(self):
        if self.D:
            return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.2
        # In: 28 x 28 x 1, depth = 1
        # Out: 14 x 14 x 1, depth=64
        input_shape = (self.img_rows, self.img_cols, self.channel)
        self.D.add(Conv2D(depth*1, 5, strides=2, input_shape=input_shape,\
            padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        self.D.summary()        #输出模型结构

        return self.D

    """生成器网络：卷积生成"""
    def generator(self):
        if self.G:
            return self.G
        self.G = Sequential()
        dropout = 0.2
        depth = 64+64+64+64
        dim = 7
        # In: 100
        # Out: dim x dim x depth
        self.G.add(Dense(dim*dim*depth, input_dim=100))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, dim, depth)))
        self.G.add(Dropout(dropout))

        # In: dim x dim x depth
        # Out: 2*dim x 2*dim x depth/2
        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))        #卷积核个数：depth/2= 128，核：5*5
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D())
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        # Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))

        self.G.summary()

        return self.G

    """判别模型"""
    def discriminator_model(self):
        if self.DM:
            return self.DM

        optimizer = RMSprop(lr=0.0002, decay=6e-8)      #RMSprop是SGD的优化版：Rms指的是当前时刻梯度项平方的期望的均方根
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])


        return self.DM

    """对抗训练模型"""
    def adversarial_model(self):
        if self.AM:
            return self.AM

        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])

        return self.AM


class dcGAN(object):
    def __init__(self):
        #定义图片的长宽+颜色通道
        self.img_rows = 28
        self.img_cols = 28
        self.channel = 1

        #调用了tensorflow自带的mnist读取方法
        self.x_train = input_data.read_data_sets('MNIST_data',\
        	one_hot=True).train.images

        #原来n*m，原来的1行m列的数据变成m行1列的一组数据，那么变形后总共拥有n行组数据，
        #-1表示自适应
        self.x_train = self.x_train.reshape(-1, self.img_rows,\
        	self.img_cols, 1).astype(np.float32)        #astype实现变量类型转换

        self.Net = Net()
        self.discriminator =  self.Net.discriminator_model()
        self.adversarial = self.Net.adversarial_model()
        self.generator = self.Net.generator()

    """训练阶段"""
    def train(self, train_steps=None, batch_size=None, save_interval=None):
        noise_input = None
        if save_interval>0:
            noise_input = np.random.uniform(-1.0, 1.0, size=[16, 100])      #正态随机

        """进入训练迭代train_steps"""
        for i in range(train_steps):
            """生成器生成图片"""
            #取得训练的真图像，数量为batch_size，用randint来随机抽取下标，这些下标指向mnist训练集55000
            images_train = self.x_train[np.random.randint(0,
                self.x_train.shape[0], size=batch_size), :, :, :]       #后面的3个冒号为28,28,1，__init__有定义
            #生成训练的假图像
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            images_fake = self.generator.predict(noise)     #用噪音生成的图像当做假图

            """判别模型判别真假图片"""
            #合并成判别模型的输入训练数据x
            x = np.concatenate((images_train, images_fake))     #拼接真图和假图，当做数据
            #生成对抗模型真假标记组y
            y = np.ones([2*batch_size, 1])      #初始化标签，2*256行，1列 ，全为1，
            y[batch_size:, :] = 0       #ba_size行开始，所有列元素变为0，相当于y标签前半段真1，后边段假0,对应x
            #判别器损失
            d_loss = self.discriminator.train_on_batch(x, y)        #本函数在一个batch的数据上进行一次参数更新，返回训练误差组（损失，准确）

            """在对抗模型中推进生成器和判别器训练 """
            #对抗模型真标记组y
            y = np.ones([batch_size, 1])
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            #对抗损失，利用adversarial模型使得生成器和判别器相互对抗
            a_loss = self.adversarial.train_on_batch(noise, y)

            log_mesg = "%d: [discriminator loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
            log_mesg = "%s  [adversarial loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
            print(log_mesg)

            if save_interval>0:
                if (i+1)%save_interval==0:      #如果到了存储间隔点，输出生成图片
                    self.plot_images(save2file=True, samples=noise_input.shape[0],\
                        noise=noise_input, step=(i+1))


    """输出阶段训练图片"""
    def plot_images(self, save2file=False, fake=True, samples=16, noise=None, step=0):
        filename = 'mnist.png'
        if fake:
            if noise is None:
                noise = np.random.uniform(-1.0, 1.0, size=[samples, 100])
            else:
                filename = "mnist_%d.png" % step        #文件命名
            images = self.generator.predict(noise)      #用训练现有阶段的生成器生成图片
        else:
            i = np.random.randint(0, self.x_train.shape[0], samples)
            images = self.x_train[i, :, :, :]

        plt.figure(figsize=(10,10))

        for i in range(images.shape[0]):
            plt.subplot(4, 4, i+1)      #绘制多个子图,(numRows, numCols, plotNum)
            image = images[i, :, :, :]
            image = np.reshape(image, [self.img_rows, self.img_cols])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
        plt.tight_layout()

        if save2file:
            plt.savefig(filename)
            plt.close('all')
        else:
            plt.show()

if __name__ == '__main__':
    timer = ElapsedTimer()
    timer.elapsed_time()

    mnist_dcgan = dcGAN()
    mnist_dcgan.train(train_steps=200, batch_size=256, save_interval=1)

    mnist_dcgan.plot_images(fake=True)
    mnist_dcgan.plot_images(fake=False, save2file=True)