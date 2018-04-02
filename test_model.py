import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as pl
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)



test_data = mnist.test.images
test_labels = mnist.test.labels

rand = np.random.randint(0,len(test_data))
image = test_data[rand]
img = test_data[rand]
img.shape = (1,784)
print(img.shape)
image.shape = (28,28)
label = test_labels[rand]   



image = np.array(image)
label = np.array(label)
for x in range(len(image)):
    image[x]*=255


input_l = 784
hidden1_l = 200
hidden2_l = 150
output_l = 10
dropout = 0.75



W1 = tf.Variable(tf.random_normal([input_l,hidden1_l]),dtype=tf.float32,name='w1')
W2 = tf.Variable(tf.random_normal([hidden1_l,hidden2_l]),dtype=tf.float32,name='w2')
W3 = tf.Variable(tf.random_normal([hidden2_l,output_l]),dtype=tf.float32,name='w3')
B1 = tf.Variable(tf.zeros(hidden1_l),dtype=tf.float32,name='b1')
B2 = tf.Variable(tf.zeros(hidden2_l),dtype=tf.float32,name='b2')
B3 = tf.Variable(tf.zeros(output_l),dtype=tf.float32,name='b3')


def feedForward(data):


    hl1 = tf.add(tf.matmul(data,W1),B1)
    hl1 = tf.nn.relu(hl1)
    hl1 = tf.nn.dropout(hl1,dropout)
    hl2 = tf.add(tf.matmul(hl1,W2),B2)
    hl2 = tf.nn.relu(hl2)
    hl2 = tf.nn.dropout(hl2, dropout)
    output = tf.add(tf.matmul(hl2,W3),B3)
    output = tf.nn.sigmoid(output)

    return output

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    saver.restore(sess,"C:\\Users\\enead\\Desktop\\Dump\Python\\savednn\\nn.ckpt")
    out = feedForward(img)
    a = tf.argmax(out,1)
    print('Value predicted by the network',sess.run(a))
 

pl.imshow(image)
pl.title('Test Image')
pl.show()
