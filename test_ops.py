import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio
from ops import *

def DM_range(a,b,name="DM_range"):
    with tf.variable_scope(name):
        dm = 0
        for i in range(a,b):
            prediction = np.array(sess.run(output, 
                                           feed_dict={x_data_tem:x_data[:,:,i:i+1],
                                                      lv_out_tem:lv_out_label[:,:,i:i+1],
                                                      keep_prob:1}))
            prediction[np.where(prediction>0.5)],prediction[np.where(prediction<=0.5)] = 1,0
            acc_y =  lv_out_label[:,:,i]
            dm = Dice_Metric(prediction,acc_y)+dm
            #print("Test",i," Dice Metric:",Dice_Metric(prediction,acc_y))
        return dm/(b-a)

#数据读入
datafile = 'F:\\卷积网络\\fcngan\\cardiac_data.mat'
data = scio.loadmat(datafile)
x_data = np.array(data['image'])#原始数据
lv_wall_label = np.array(data['lv_wall'])#内外膜之间的心肌层
lv_in_label = np.array(data['lv_in'])#左心室心内膜
lv_out_label = np.array(data['lv_out'])#左心室心外膜
index_label = np.array(data['cardiac_index'])#指数的标签
#超参数
batch = 1#每次送入图片大小
conv_dim = 4#卷积核个数
keep_prob = tf.placeholder(tf.float32)
#变量
x_data_tem = tf.placeholder(tf.float32,[80,80,batch])#标签和数据首先被送到这里
x_data_input = tf.reshape(x_data_tem,[batch,80,80,1])#需要改变送来数据的形状，这是原始图像
lv_out_tem = tf.placeholder(tf.float32, [80,80,batch])#同上,标签被送到这里
lv_out_input = tf.reshape(lv_out_tem,[80,80])#这是左心室心外膜
lv_in_tem = tf.placeholder(tf.float32, [80,80,batch])#同上,标签被送到这里
lv_in_input = tf.reshape(lv_in_tem,[80,80])#这是左心室心内膜

output = Fcc_test(x_data_input=x_data_input,conv_dim=conv_dim,
                  batch=batch,keep_prob=keep_prob,name='fcc_net')
#output = U_net(x_input=x_data_input,conv_dim=32,batch=1,keep_prob=keep_prob,name='U_net')
#output = Fcc_test2(x_data_input,conv_dim=4,batch=1,keep_prob=0.5,name='fcc_net2')
#<tf.Tensor 'Reshape_4:0' shape=(80, 80) dtype=float32>

#损失函数，梯度下降优化器

loss = tf.reduce_mean(tf.abs(lv_out_input-output))#平均值
#loss = tf.reduce_mean(tf.square(lv_in_input-output))#平均值
#tf.reduce_mean(tf.abs(label_input - output1))

#train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)
'''
Test 0  DM_range: 0.907524172340595
Test 1  DM_range: 0.9134896327694446
Test 2  DM_range: 0.9256185951652259
Test 3  DM_range: 0.9295725857337462
Test 4  DM_range: 0.9346079645969916
Test 5  DM_range: 0.9327518428163648
Test 6  DM_range: 0.9344523035782059
Test 7  DM_range: 0.9359432425292541
Test 8  DM_range: 0.9365714145825613
Test 9  DM_range: 0.9332788338312573
'''
train_step = tf.train.AdamOptimizer(0.0002,beta1=0.5).minimize(loss)
'''
Test 0  DM_range: 0.8703519787741336
Test 1  DM_range: 0.8871851732959323
Test 2  DM_range: 0.9138472784032543
Test 3  DM_range: 0.9219695298464855
Test 4  DM_range: 0.9242898876814791
Test 5  DM_range: 0.9317091019823989
Test 6  DM_range: 0.9354722550569236
Test 7  DM_range: 0.9402734678857518
Test 8  DM_range: 0.9378064530666705
Test 9  DM_range: 0.9389837398173537

Test 0  DM_range: 0.9402901301480087
Test 1  DM_range: 0.9424243584552738
Test 2  DM_range: 0.9419480128813963
Test 3  DM_range: 0.9421532955331428
Test 4  DM_range: 0.9395880890212959
还未收敛
'''
#train_step = tf.train.AdadeltaOptimizer(1).minimize(loss)
'''
Test 0  DM_range: 0.9116915690618838
Test 1  DM_range: 0.9137260680299903
Test 2  DM_range: 0.9269538477499161
Test 3  DM_range: 0.9320744616584455
Test 4  DM_range: 0.9332072319450793
Test 5  DM_range: 0.9358735635874041
Test 6  DM_range: 0.9382213980112364
Test 7  DM_range: 0.9400485702513499
Test 8  DM_range: 0.940939632437682
Test 9  DM_range: 0.9410387664628833
===
Test 0  DM_range: 0.8951141603750344
Test 1  DM_range: 0.9178540977313382
Test 2  DM_range: 0.922752064119638
Test 3  DM_range: 0.9340422734842617
Test 4  DM_range: 0.9356942851455698
Test 5  DM_range: 0.934867350508256
Test 6  DM_range: 0.9381638869449156
Test 7  DM_range: 0.9389763706586559
Test 8  DM_range: 0.9400012178542346
Test 9  DM_range: 0.9402597949620789
Test 10  DM_range: 0.940360620372749
Test 11  DM_range: 0.9393898550008583
Test 12  DM_range: 0.9399304090563333
Test 13  DM_range: 0.9399704673636458
Test 14  DM_range: 0.9392242787760492
'''
#train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9, 
#                                        use_nesterov=True).minimize(loss)
'''
Test 0  DM_range: 0.8750959992091112
Test 1  DM_range: 0.9226574402400297
Test 2  DM_range: 0.9340580274312831
Test 3  DM_range: 0.9256257787975009
Test 4  DM_range: 0.903551136003487
Test 5  DM_range: 0.9195298122910783
Test 6  DM_range: 0.9180601696333305
Test 7  DM_range: 0.9288878243419335
Test 8  DM_range: 0.9348483810470856
Test 9  DM_range: 0.9312046361527788
'''
#train_step = tf.train.AdagradOptimizer(0.01).minimize(loss)
'''
Test 0  DM_range: 0.7876549600584768
Test 1  DM_range: 0.8716823733609588
Test 2  DM_range: 0.881182320680282
Test 3  DM_range: 0.8954051255954513
Test 4  DM_range: 0.8961051302005265
Test 5  DM_range: 0.9019132771546162
Test 6  DM_range: 0.9084558632103954
Test 7  DM_range: 0.9124364788441877
Test 8  DM_range: 0.9110847910238586
Test 9  DM_range: 0.9157920377439323
'''
#train_step = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(loss)
'''
Test 0  DM_range: 0.8934191151171894
Test 1  DM_range: 0.9172087821953447
Test 2  DM_range: 0.9184446866293818
Test 3  DM_range: 0.9224149753170376
Test 4  DM_range: 0.9195132068746614
Test 5  DM_range: 0.9157706189746346
Test 6  DM_range: 0.9122588105401754
Test 7  DM_range: 0.9103484901215211
Test 8  DM_range: 0.917033083101782
Test 9  DM_range: 0.9239424830385167
'''
#train_step = tf.train.RMSPropOptimizer(0.01).minimize(loss)
'''
Test 0  DM_range: 0.9153069933261271
Test 1  DM_range: 0.9168890050355517
Test 2  DM_range: 0.9297069237479831
Test 3  DM_range: 0.924683483415422
Test 4  DM_range: 0.9245984060761995
Test 5  DM_range: 0.9287949684316011
Test 6  DM_range: 0.9351222979113797
Test 7  DM_range: 0.9342188941486326
Test 8  DM_range: 0.9380549350667962
Test 9  DM_range: 0.9368616060389314
'''
#https://blog.csdn.net/weixin_40170902/article/details/80092628
#训练初始化
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#训练
test_or = 2800

for i in range(15):
    for j in range(test_or):
        sess.run(train_step, 
                 feed_dict={x_data_tem:x_data[:,:,j:j+1],lv_out_tem:lv_out_label[:,:,j:j+1],keep_prob:1})
    print("Test",i," DM_range:",DM_range(2800,2899,name='dm3'))

    prediction = np.array(sess.run(output, feed_dict={x_data_tem:x_data[:,:,test_or:test_or+1],
                                                 lv_out_tem:lv_out_label[:,:,test_or:test_or+1]
                                                 ,keep_prob:1}))
    prediction[np.where(prediction>0.5)],prediction[np.where(prediction<=0.5)] = 1,0
    plt.figure()
    plt.subplot(121)
    plt.imshow(prediction,cmap ='gray')
    plt.subplot(122)
    acc_y = lv_out_label[:,:,test_or]
    plt.imshow(acc_y,cmap ='gray')

def train(output,lv_out_input,name="train"):
    with tf.variable_scope(name):
        loss = tf.reduce_mean(tf.abs(lv_out_input-output))#平均值
        train_step = tf.train.AdamOptimizer(0.0002,beta1=0.5).minimize(loss)
        return train_step

for patient in range(0,145):
    '''
    第0个病人      20:2899
    第1个病人 0:19 40:2899
    第2个病人 0:39 60:2899
    第3个病人 0:59 80:2899
    0:patient*20-1 (patient+1)*20:2899
    
    lv_out = tf.placeholder(tf.float32, [80,80])
    lv_in  = tf.placeholder(tf.float32, [80,80])
    
    for i in range(0,patient*20):
        sess.run(train_step,
                     feed_dict={x_data_tem:x_data[:,:,i],lv_out:lv_out_label[:,:,i],keep_prob:1})
    for i in range((patient+1)*20,2900):
        sess.run(train_step,
                     feed_dict={x_data_tem:x_data[:,:,i],lv_out:lv_out_label[:,:,i],keep_prob:1})
    '''
    for i in range(15):
        for j in range(test_or):
            sess.run(train_step, 
                     feed_dict={x_data_tem:x_data[:,:,j:j+1],lv_out_tem:lv_out_label[:,:,j:j+1],keep_prob:1})
'''
for i in range(1,10):
    print(i)
1 9
'''
'''
x_data = np.array(data['image'])#原始数据
lv_wall_label = np.array(data['lv_wall'])#内外膜之间的心肌层
lv_in_label = np.array(data['lv_in'])#左心室心内膜
lv_out_label = np.array(data['lv_out'])#左心室心外膜
index_label = np.array(data['cardiac_index'])#指数的标签
#超参数
batch = 1#每次送入图片大小
conv_dim = 4#卷积核个数
#变量
x_data_tem = tf.placeholder(tf.float32,[80,80,batch])#标签和数据首先被送到这里
x_data_input = tf.reshape(x_data_tem,[batch,80,80,1])#需要改变送来数据的形状，这是原始图像
lv_out_tem = tf.placeholder(tf.float32, [80,80,batch])#同上,标签被送到这里
lv_out_input = tf.reshape(lv_out_tem,[80,80])#这是左心室心外膜
'''

























