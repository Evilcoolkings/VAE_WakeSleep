from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import math
from util import plot, plot_loss, get_normal_prob, get_sample_z

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
X_dim = mnist.train.images.shape[1]
Hidden_dim = 512
Z_dim = 2
batch_size = 512
X_ph = tf.placeholder(tf.float64, shape=[None, X_dim])
Z_ph = tf.placeholder(tf.float64, shape=[None, Z_dim])
K_EVALUATION = 1000
NUM_EPOCH = 100
TRAIN_DATA_SIZE = 60000
TEST_DATA_SIZE = 10000
Total_Train_Step = int(math.ceil(TRAIN_DATA_SIZE/batch_size))*NUM_EPOCH
Train_Epoch_Step = int(math.ceil(TRAIN_DATA_SIZE/batch_size))
Test_Epoch_Step = int(math.ceil(TEST_DATA_SIZE/batch_size))

'''
Decoding part
'''
with tf.variable_scope("Decode"):
    P_W1 = tf.Variable(tf.random_normal([Z_dim, Hidden_dim], stddev=0.01, dtype=tf.float64), name='P_W1')
    P_b1 = tf.Variable(tf.zeros([Hidden_dim], dtype=tf.float64), name='P_b1')
    P_W2 = tf.Variable(tf.random_normal([Hidden_dim, X_dim], stddev=0.01,dtype=tf.float64), name='P_W2')
    P_b2 = tf.Variable(tf.zeros([X_dim],dtype=tf.float64), name='P_b2')

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
with tf.variable_scope("Encode"):
    Q_W1 = tf.Variable(tf.random_normal([X_dim, Hidden_dim], stddev=0.01,dtype=tf.float64), name='Q_W1')
    Q_b1 = tf.Variable(tf.zeros([Hidden_dim], dtype=tf.float64), name='Q_b1')
    Q_W2_mu = tf.Variable(tf.random_normal([Hidden_dim, Z_dim], stddev=0.01, dtype=tf.float64), name='Q_W2_mu')
    Q_b2_mu = tf.Variable(tf.zeros([Z_dim], dtype=tf.float64), name='Q_b2_mu')
    Q_W2_var = tf.Variable(tf.random_normal([Hidden_dim, Z_dim], stddev=0.01, dtype=tf.float64), name='Q_W2_var')
    Q_b2_var = tf.Variable(tf.zeros([Z_dim], dtype=tf.float64), name='Q_b2_var')

def encoder_network(X):
    '''
    Prepare decoder network Q(z|X) 
    '''
    hidden_layer = tf.nn.relu(tf.matmul(X, Q_W1)+Q_b1)

    Z_mu = tf.matmul(hidden_layer, Q_W2_mu)+Q_b2_mu

    Z_var = tf.matmul(hidden_layer, Q_W2_var)+Q_b2_var

    return Z_mu, Z_var

'''
Wake Phase
'''
def get_Wake_Phase_loss(X):
    # sample z from Q(z|x)
    Z_mu, Z_var = encoder_network(X)
    Z_sample, _ = get_sample_z(Z_mu, Z_var)
    _, logits = dencoder_network(Z_sample)

    # -E[log P(X|Z)] so we need to minimize the following term
    E_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(targets=X, logits=logits), 1)

    return E_loss

def get_Wake_Phase_solver(loss):
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Decode')
    solver = tf.train.AdamOptimizer().minimize(loss, var_list= train_vars)
    return solver



'''
Sleep Phase
'''
def get_pxz_samples(sess):
    Z_samples_var = np.random.randn(batch_size, Z_dim)
    feed_dict = {
        Z_ph: Z_samples_var
    }
    X_samples, _ = dencoder_network(Z_ph)
    X_samples_var = sess.run(X_samples, feed_dict = feed_dict )
    X_samples_var = np.random.binomial(1, p=X_samples_var)

    return X_samples_var, Z_samples_var

def get_Sleep_Phase_loss(X, Z):
    Z_mu, Z_logvar = encoder_network(X)
    Z_var = tf.exp(0.5 * Z_logvar)
    loss = tf.reduce_sum(0.5 * (Z - Z_mu) ** 2 / (Z_var ** 2) + Z_logvar)
    return loss

def get_Sleep_Phase_solver(loss):
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Encode')
    solver = tf.train.AdamOptimizer().minimize(loss, var_list= train_vars)
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
    fig.savefig("Wake_Sleep.png")

'''
Evaluate part
'''
def evaluate_kernel(X):
    Z_mu, Z_var = encoder_network(X)
    Z_sample, prob_q = get_sample_z(Z_mu, Z_var)
    prob, logits = dencoder_network(Z_sample)
    prob_p_xz = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(targets=X, logits=logits), 1)
    prob_p_xz = -tf.reshape(prob_p_xz, [-1,1])
    prob_p_xz = tf.exp(prob_p_xz)
    prob_p_z = get_normal_prob(Z_sample)
    tmp = prob_p_xz*prob_p_z / prob_q
    return tmp

def evaluate(sess, X):

    one_step_eval= evaluate_kernel(X)
    trainresult = []
    for i in range(Train_Epoch_Step):
        print("Temp i:" + str(i))
        feed_dict = {
            X: mnist.train.next_batch(batch_size)[0]
        }
        tmpresult = []
        for k in range(K_EVALUATION):
            tmp = sess.run([one_step_eval], feed_dict=feed_dict)
            tmpresult.append(tmp)
        tmpresult = np.mean(np.asarray(tmpresult),0)
        tmpresult = np.log(tmpresult)
        trainresult.append(np.mean(tmpresult))

    testresult = []
    for i in range(Test_Epoch_Step):
        feed_dict = {
            X: mnist.test.next_batch(batch_size)[0]
        }
        tmpresult = []
        for k in range(K_EVALUATION):
            tmp= sess.run([one_step_eval], feed_dict=feed_dict)
            tmpresult.append(tmp)
        tmpresult = np.mean(np.asarray(tmpresult),0)
        tmpresult = np.log(tmpresult)
        testresult.append(np.mean(tmpresult))
    return np.mean(trainresult), np.mean(testresult)


def main():

    # Get wake solver
    wake_loss = get_Wake_Phase_loss(X_ph)
    wake_solver = get_Wake_Phase_solver(wake_loss)
    # Get sleep solver
    sleep_loss = get_Sleep_Phase_loss(X_ph, Z_ph)
    sleep_solver = get_Sleep_Phase_solver(sleep_loss)


    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_loss = []
    test_loss = []

    for i in range(Total_Train_Step):
        if i % Train_Epoch_Step == 0:
            print(i)
        if i % (10*Train_Epoch_Step) == 0:
            print("<<<<<<<<<<<<<<evaluation>>>>>>>>>>>>>>")
            print("Step:"+str(i))
            train_L, test_L = evaluate(sess, X_ph)
            train_loss.append(train_L)
            test_loss.append(test_L)
            print('train_L: %f, test_L: %f'%(train_L,test_L))
        '''
        Wake phase
        '''
        feed_dict = {
            X_ph: mnist.train.next_batch(batch_size)[0]
        }
        sess.run(wake_solver, feed_dict=feed_dict)

        '''
        Sleep phase
        '''
        X_samples_var, Z_samples_var = get_pxz_samples(sess)
        feed_dict = {
            X_ph: X_samples_var,
            Z_ph: Z_samples_var,
        }
        sess.run(sleep_solver, feed_dict=feed_dict)

    #Final evaluation
    print("<<<<<<<<<<<<<<evaluation>>>>>>>>>>>>>>")
    print("Step:" + str(i))
    train_L, test_L = evaluate(sess, X_ph)
    train_loss.append(train_L)
    test_loss.append(test_L)
    print('train_L: %f, test_L: %f' % (train_L, test_L))
    #plot result
    plot_result(sess)
    plot_loss(train_loss, test_loss, 'WakeSleep_loss')
    # save model
    save_path = saver.save(sess, "Model/ws_model.ckpt")



if __name__ == "__main__":
    main()