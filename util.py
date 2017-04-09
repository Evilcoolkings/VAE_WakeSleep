import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
import numpy as np
import math

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def plot_loss(train_loss, test_loss, name):
    fig = plt.figure()
    plt.plot(train_loss,"g-",label="train_L")
    plt.plot(test_loss,"r-.",label="test_L")

    plt.xlabel("10 epoches")
    plt.title(name)

    plt.grid(True)
    plt.legend()
    fig.savefig(name+".png")


def get_normal_prob(eps):
    prob = tf.exp(-0.5*eps**2)/(tf.sqrt(2*np.float64(math.pi)))
    prob = prob[:,0]*prob[:,1]
    prob = tf.reshape(prob, [-1,1])
    return prob

def get_sample_z(mu, var):
    # TODO var or log_var
    eps = tf.random_normal(shape=tf.shape(mu), dtype=tf.float64)
    sample =  mu + tf.exp(var / 2) * eps
    prob = get_normal_prob(eps)
    return sample, prob

