Variational Autoencoders
========================

It contains two algorithms: Wake-Sleep algorithm and Autoencoding variational Bayes.

You can see report.pdf to know the details of these two algorithm.

Need tensorflow version 0.11


When you want to run the model, just use the following command:

Python WakeSleep.py

Python VAE.py

It will train the model, evaluate by L1000, generate random samples, draw X,Z distribution.

Implementation and Experience
=============================

L_1000 of both model
--------------------

![image](https://github.com/cmusjtuliuyuan/VAE_WakeSleep/blob/master/Sleep_Wake_loss.png) 
![image](https://github.com/cmusjtuliuyuan/VAE_WakeSleep/blob/master/VAE_loss.png)

Visualize of both algorithm.The left one is about sleep wake algorithm, and the right part is about AEVB algorithm
------------------------------------------------------------------------------------------------------------------
![image](https://github.com/cmusjtuliuyuan/VAE_WakeSleep/blob/master/Wake_Sleep.png)
![image](https://github.com/cmusjtuliuyuan/VAE_WakeSleep/blob/master/VAE.png)

