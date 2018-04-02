import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import _pickle as pickle

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

#setting learning variables
learning_rate = 0.5
epochs = 1100
batch = 550

#setting network variables
input_l = 784
hidden1_l = 200
hidden2_l = 150
output_l = 10
dropout = 0.75


#getting data ready
training_data = mnist.train._images
training_labels = mnist.train._labels
testing_data = mnist.test._images
testing_labels = mnist.test._labels



#building the network
    #input and  output layers
x = tf.placeholder(tf.float32, [None,input_l])
y = tf.placeholder(tf.float32, [None,output_l])
    #weights and biases
W1 = tf.Variable(tf.random_normal([input_l,hidden1_l]),dtype=tf.float32,name='w1')
W2 = tf.Variable(tf.random_normal([hidden1_l,hidden2_l]),dtype=tf.float32,name='w2')
W3 = tf.Variable(tf.random_normal([hidden2_l,output_l]),dtype=tf.float32,name='w3')
B1 = tf.Variable(tf.zeros([hidden1_l]),dtype=tf.float32,name='b1')
B2 = tf.Variable(tf.zeros([hidden2_l]),dtype=tf.float32,name='b2')
B3 = tf.Variable(tf.zeros([output_l]),dtype=tf.float32,name='b3')


saver = tf.train.Saver([W1,W2,W3,B1,B2,B3])

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

def train(x):
    prediction = feedForward(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits =prediction,labels=y))
    optimiser = tf.train.AdamOptimizer(0.001).minimize(cost)
    #0.001
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(epochs):
            epoch_loss = 0

            for _ in range(int(len(training_data)/batch)):
                e_x,e_y = mnist.train.next_batch(batch)
                _,e_c = sess.run([optimiser,cost],feed_dict={x: e_x, y: e_y})
                epoch_loss += e_c
                print(epoch_loss)
            print('Epoch', epoch, 'completed out of ', epochs, 'loss: ', epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        acc = accuracy.eval({x:testing_data, y: testing_labels})
        print('Accuracy:', acc)
        if(acc>=0.96):
            save_path = saver.save(sess,"C:\\Users\\enead\\Desktop\\Dump\Python\\savednn\\nn.ckpt")
            print("Saved at: ",save_path)

train(x)
