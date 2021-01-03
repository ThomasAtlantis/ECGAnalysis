import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA

dataset = np.array([
    [np.load(f"train_{i}.npy"), np.load(f"valid_{i}.npy")] 
    for i in range(process_data.FOLD)
], dtype=object)




C = 200  # 样本数
x = np.arange(C)
s1 = 2 * np.sin(0.02 * np.pi * x)  # 正弦信号
 
a = np.linspace(-2, 2, 25)
s2 = np.array([a, a, a, a, a, a, a, a]).reshape(200, )  # 锯齿信号
s3 = np.array(20 * (5 * [2] + 5 * [-2]))  # 方波信号
s4 = 4 * (np.random.random([1, C]) - 0.5).reshape(200, ) #随机信号
"""画出4种信号"""
ax1 = plt.subplot(411)
ax2 = plt.subplot(412)
ax3 = plt.subplot(413)
ax4 = plt.subplot(414)
ax1.plot(x,s1)
ax2.plot(x,s2)
ax3.plot(x,s3)
ax4.plot(x,s4)
plt.show()
"""将4种信号混合
其中mix矩阵是混合后的信号矩阵，shape=[4,200]"""
s=np.array([s1,s2,s3,s4])
ran=2*np.random.random([4,4])
mix=ran.dot(s)
ax1 = plt.subplot(411)
ax2 = plt.subplot(412)
ax3 = plt.subplot(413)
ax4 = plt.subplot(414)
ax1.plot(x,mix[0,:])
ax2.plot(x,mix[1,:])
ax3.plot(x,mix[2,:])
ax4.plot(x,mix[3,:])
plt.show()
 
ica = FastICA(n_components=5) #独立成分为4个
mix = mix.T #将信号矩阵转为[n_samples,n_features],即[200,4]
u = ica.fit_transform(mix) # u为解混后的4个独立成分，shape=[200,4]
u = u.T # shape=[4,200]
print(u.shape)
print(ica.n_iter_) # 算法迭代次数
"""画出解混后的4种独立成分"""
ax1 = plt.subplot(511)
ax2 = plt.subplot(512)
ax3 = plt.subplot(513)
ax4 = plt.subplot(514)
ax5 = plt.subplot(515)
ax1.plot(x,u[0,:])
ax2.plot(x,u[1,:])
ax3.plot(x,u[2,:])
ax4.plot(x,u[3,:])
ax5.plot(x,u[4,:])
plt.show()