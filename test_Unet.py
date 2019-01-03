import tensorflow as tf
import numpy as np


def batch_norm(in_image,epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    with tf.variable_scope(name):
        return tf.contrib.layers.batch_norm(in_image, decay=momentum, 
                                            updates_collections=None, epsilon=epsilon,
                                            scale=True, scope=name)
        
def group_norm(x, G=32, epsilon=1e-5, name="group_norm"):
    # x: input features with shape [N,C,H,W]
    # gamma, beta: scale and offset, with shape [1,C,1,1]
    # G: number of groups for GN
    with tf.variable_scope(name):
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [N, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + epsilon)
        # per channel gamma and beta
        gamma = tf.get_variable('gamma', [C],initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [C],initializer=tf.constant_initializer(0.0))
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])
        output = tf.reshape(x, [N, C, H, W]) * gamma + beta
        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])
        return output

def norm(x,norm="group",name="normalization"):
    with tf.variable_scope(name):
        if norm == "null":
            return x
        if norm == "group":
            return group_norm(x,name=name)
        if norm == "batch":
            return batch_norm(x,name=name)

def conv2d(in_image, out_dim, w_row=5, w_col=5, strides_row=2, strides_col=2, stddev=0.02, 
           pad='SAME',name="conv2d"):
    with tf.variable_scope(name):
        w    = tf.get_variable('w', [w_row, w_col, in_image.get_shape()[-1], out_dim], 
                                     initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(in_image, w, strides=[1, strides_row, strides_col, 1], padding=pad)
        b    = tf.get_variable('b', [out_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())
        return conv
    
def deconv2d(in_image, output_shape, w_row=5, w_col=5, strides_row=2, 
             strides_col=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):#output_shape [batch,80,80,dim]
        w = tf.get_variable('w', [w_row, w_col, output_shape[-1], in_image.get_shape()[-1]],
                                  initializer=tf.random_normal_initializer(stddev=stddev))
        try:
            deconv = tf.nn.conv2d_transpose(in_image, w, output_shape=output_shape,
                                            strides=[1, strides_row, strides_col, 1])
        except AttributeError:
            deconv = tf.nn.deconv2d(in_image, w, output_shape=output_shape,
                                strides=[1, strides_row, strides_col, 1])
        b      = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())
        return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)


def Dice_Metric(image1,image2):
    with tf.variable_scope("DM"):
        row,col = image1.shape
        return np.cumsum(np.logical_and(image1,image2))[row*col-1]/np.cumsum(np.logical_or(image1
                        ,image2))[row*col-1]
    

    
def U_net(x_input,conv_dim=32,batch=1,keep_prob=0.5,name='U_net'):
    with tf.variable_scope(name):
        #*****************************************************
        #x_input:1*80*80*1
        conv1 = conv2d(x_input, conv_dim, name='conv1')
        #conv1:1*40*40*32
        conv2 = norm(conv2d(lrelu(conv1), conv_dim*2, name='conv2'), name='n_conv2')
        #conv1:1*20*20*64
        conv3 = norm(conv2d(lrelu(conv2), conv_dim*4, name='conv3'), name='n_conv3')
        #conv3:1*10*10*128-
        conv4 = norm(conv2d(lrelu(conv3), conv_dim*8, name='conv4'), name='n_conv4')
        #conv4:1*5*5*256-
        conv5 = norm(conv2d(lrelu(conv4), conv_dim*8, name='conv5'), name='n_conv5')
        #conv4:1*3*3*256-
        conv6 = norm(conv2d(lrelu(conv5), conv_dim*8, name='conv6'), name='n_conv6')
        #conv4:1*2*2*256-
        conv7 = norm(conv2d(lrelu(conv6), conv_dim*8, name='conv7'), name='n_conv7')
        #conv4:1*1*1*256-
        #*****************************************************
        dec1 = deconv2d(tf.nn.relu(conv7),[batch, 2, 2, conv_dim*8], name='dec1')
        dec1_dro = tf.nn.dropout(norm(dec1,name='n_dec1'), keep_prob)
        dec1_out = tf.concat([dec1_dro, conv6], 3)
        #dec1_dro:1*2*2*512
        
        dec2 = deconv2d(tf.nn.relu(dec1_out),[batch, 3, 3, conv_dim*8], name='dec2')
        dec2_dro = tf.nn.dropout(norm(dec2,name='n_dec2'), keep_prob)
        dec2_out = tf.concat([dec2_dro, conv5], 3)
        #dec1_dro:1*3*3*512
        
        dec3 = deconv2d(tf.nn.relu(dec2_out),[batch, 5, 5, conv_dim*8], name='dec3')
        dec3_n = norm(dec3,name='n_dec3')
        dec3_out = tf.concat([dec3_n, conv4], 3)
        #dec1_dro:1*5*5*512
        
        dec4 = deconv2d(tf.nn.relu(dec3_out),[batch, 10, 10, conv_dim*4], name='dec4')
        dec4_n = norm(dec4,name='n_dec4')
        dec4_out = tf.concat([dec4_n, conv3], 3)
        #dec1_dro:1*10*10*256
        
        dec5 = deconv2d(tf.nn.relu(dec4_out),[batch, 20, 20, conv_dim*2], name='dec5')
        dec5_n = norm(dec5,name='n_dec5')
        dec5_out = tf.concat([dec5_n, conv2], 3)
        #dec1_dro:1*20*20*128
        
        dec6 = deconv2d(tf.nn.relu(dec5_out),[batch, 40, 40, conv_dim], name='dec6')
        dec6_n = norm(dec6,name='n_dec6')
        dec6_out = tf.concat([dec6_n, conv1], 3)
        #dec1_dro:1*40*40*64
        
        dec7 = deconv2d(tf.nn.relu(dec6_out),[batch, 80, 80, 1], name='dec7')
        #dec1_dro:1*80*80*1
        
        return tf.reshape(tf.nn.tanh(dec7),[80,80])
    
def U_net_cat(x_input,conv_dim=32,batch=1,keep_prob=0.5,name='U_net'):
    with tf.variable_scope(name):
        #*****************************************************
        #x_input:1*80*80*1
        conv1 = conv2d(x_input, conv_dim, name='conv1')
        #conv1:1*40*40*32
        conv2 = norm(conv2d(lrelu(conv1), conv_dim*2, name='conv2'), name='n_conv2')
        #conv1:1*20*20*64
        conv3 = norm(conv2d(lrelu(conv2), conv_dim*4, name='conv3'), name='n_conv3')
        #conv3:1*10*10*128-
        conv4 = norm(conv2d(lrelu(conv3), conv_dim*8, name='conv4'), name='n_conv4')
        #conv4:1*5*5*256-
        conv5 = norm(conv2d(lrelu(conv4), conv_dim*8, name='conv5'), name='n_conv5')
        #conv4:1*3*3*256-
        conv6 = norm(conv2d(lrelu(conv5), conv_dim*8, name='conv6'), name='n_conv6')
        #conv4:1*2*2*256-
        conv7 = norm(conv2d(lrelu(conv6), conv_dim*8, name='conv7'), name='n_conv7')
        #conv4:1*1*1*256-
        #*****************************************************
        dec1 = deconv2d(tf.nn.relu(conv7),[batch, 2, 2, conv_dim*8], name='dec1')
        dec1_dro = tf.nn.dropout(norm(dec1,name='n_dec1'), keep_prob)
        conv6_cat = norm(conv2d(lrelu(conv6), conv_dim*8,w_row=3, w_col=3,
                                strides_row=1, strides_col=1, name='conv6_cat'), name='n_conv6_cat')
        dec1_out = tf.concat([dec1_dro, conv6_cat], 3)
        #dec1_dro:1*2*2*512
        
        dec2 = deconv2d(tf.nn.relu(dec1_out),[batch, 3, 3, conv_dim*8], name='dec2')
        dec2_dro = tf.nn.dropout(norm(dec2,name='n_dec2'), keep_prob)
        conv5_cat = norm(conv2d(lrelu(conv5), conv_dim*8,w_row=3, w_col=3,
                                strides_row=1, strides_col=1, name='conv5_cat'), name='n_conv5_cat')
        dec2_out = tf.concat([dec2_dro, conv5_cat], 3)
        #dec1_dro:1*3*3*512
        
        dec3 = deconv2d(tf.nn.relu(dec2_out),[batch, 5, 5, conv_dim*8], name='dec3')
        dec3_n = norm(dec3,name='n_dec3')
        conv4_cat = norm(conv2d(lrelu(conv4), conv_dim*8,w_row=3, w_col=3,
                                strides_row=1, strides_col=1, name='conv4_cat'), name='n_conv4_cat')
        dec3_out = tf.concat([dec3_n, conv4_cat], 3)
        #dec1_dro:1*5*5*512
        
        dec4 = deconv2d(tf.nn.relu(dec3_out),[batch, 10, 10, conv_dim*4], name='dec4')
        dec4_n = norm(dec4,name='n_dec4')
        conv3_cat = norm(conv2d(lrelu(conv3), conv_dim*4,w_row=3, w_col=3,
                                strides_row=1, strides_col=1, name='conv3_cat'), name='n_conv3_cat')
        dec4_out = tf.concat([dec4_n, conv3_cat], 3)
        #dec1_dro:1*10*10*256
        
        dec5 = deconv2d(tf.nn.relu(dec4_out),[batch, 20, 20, conv_dim*2], name='dec5')
        dec5_n = norm(dec5,name='n_dec5')
        conv2_cat = norm(conv2d(lrelu(conv2), conv_dim*2,w_row=3, w_col=3,
                                strides_row=1, strides_col=1, name='conv2_cat'), name='n_conv2_cat')
        dec5_out = tf.concat([dec5_n, conv2_cat], 3)
        #dec1_dro:1*20*20*128
        
        dec6 = deconv2d(tf.nn.relu(dec5_out),[batch, 40, 40, conv_dim], name='dec6')
        dec6_n = norm(dec6,name='n_dec6')
        conv1_cat = norm(conv2d(lrelu(conv1), conv_dim,w_row=3, w_col=3,
                                strides_row=1, strides_col=1, name='conv1_cat'), name='n_conv1_cat')
        dec6_out = tf.concat([dec6_n, conv1_cat], 3)
        #dec1_dro:1*40*40*64
        
        dec7 = deconv2d(tf.nn.relu(dec6_out),[batch, 80, 80, 1], name='dec7')
        #dec1_dro:1*80*80*1
        
        return tf.reshape(tf.nn.tanh(dec7),[80,80])