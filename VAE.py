from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from util import plot

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_dim = mnist.train.images.shape[1]
Hidden_dim = 512
Z_dim = 2
batch_size = 32
X_ph = tf.placeholder(tf.float32, shape=[None, X_dim])
Z_ph = tf.placeholder(tf.float32, shape=[None, Z_dim])


'''
Decoding part
'''
P_W1 = tf.Variable(tf.random_normal([Z_dim, Hidden_dim], stddev=0.01), name='P_W1')
P_b1 = tf.Variable(tf.zeros([Hidden_dim]), name='P_b1')
P_W2 = tf.Variable(tf.random_normal([Hidden_dim, X_dim], stddev=0.01), name='P_W2')
P_b2 = tf.Variable(tf.zeros([X_dim]), name='P_b2')

def dencoder_network(Z):
    '''
    Prepare encoder network P(X|Z) 
    '''
    hidden_layer = tf.nn.relu(tf.matmul(Z, P_W1)+P_b1)

    logits = tf.matmul(hidden_layer, P_W2)+P_b2
    prob = tf.nn.sigmoid(logits, name = 'P_X_given_z')
    return prob, logits

'''
Encoding part
'''
Q_W1 = tf.Variable(tf.random_normal([X_dim, Hidden_dim], stddev=0.01), name='Q_W1')
Q_b1 = tf.Variable(tf.zeros([Hidden_dim]), name='Q_b1')
Q_W2_mu = tf.Variable(tf.random_normal([Hidden_dim, Z_dim], stddev=0.01), name='Q_W2_mu')
Q_b2_mu = tf.Variable(tf.zeros([Z_dim]), name='Q_b2_mu')
Q_W2_var = tf.Variable(tf.random_normal([Hidden_dim, Z_dim], stddev=0.01), name='Q_W2_var')
Q_b2_var = tf.Variable(tf.zeros([Z_dim]), name='Q_b2_var')

def encoder_network(X):
    '''
    Prepare decoder network Q(z|X) 
    '''
    hidden_layer = tf.nn.relu(tf.matmul(X, Q_W1)+Q_b1)

    Z_mu = tf.matmul(hidden_layer, Q_W2_mu)+Q_b2_mu

    Z_var = tf.matmul(hidden_layer, Q_W2_var)+Q_b2_var

    return Z_mu, Z_var

def get_sample_z(mu, var):
    # TODO var or log_var
    eps = tf.random_normal(shape=tf.shape(mu))
    sample =  mu + tf.exp(var / 2) * eps
    return sample

'''
Training part
'''
def get_loss(X):
    # sample z from Q(z|x)
    Z_mu, Z_var = encoder_network(X)
    Z_sample = get_sample_z(Z_mu, Z_var)

    _, logits = dencoder_network(Z_sample)

    # E[log P(X|Z)]
    E_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(targets=X, logits=logits), 1)
    # D_KL(Q(z|X) || P(z|X))
    KL_loss = 0.5 * tf.reduce_sum(tf.exp(Z_var) + Z_mu ** 2 - 1. - Z_var, 1)
    # VAE loss
    vae_loss = tf.reduce_mean(E_loss + KL_loss)
    return vae_loss

def get_solver(loss):
    solver = tf.train.AdamOptimizer().minimize(loss)
    return solver

'''
Plot part
'''
def plot_result(sess):
    feed_dict ={
        Z_ph:np.random.randn(100, Z_dim)
    }
    X_samples, _ = dencoder_network(Z_ph)
    samples = sess.run(X_samples, feed_dict=feed_dict)
    fig = plot(samples)
    fig.savefig("VAE.png")


def main():

    loss = get_loss(X_ph)
    solver = get_solver(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(100000):
        if i % 100 == 0:
            print(i)

        feed_dict={
            X_ph:mnist.train.next_batch(batch_size)[0]
        }
        sess.run(solver, feed_dict = feed_dict)

    plot_result(sess)

if __name__ == "__main__":
    main()