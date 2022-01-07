# duralava
Duralava is a neural network which can learn to simulate a lava lamp in an infinite loop. 

## Example
![out](https://user-images.githubusercontent.com/1943719/148611813-b7da9a29-1b09-413a-965d-c6b65e79a058.png)

## How it works

Generative Adversarial Networks (GANs) can learn to generate new samples of data. For example, a GAN can be trained to output images of a lava lamp which look as real as possible. To accomplish this, the GAN gets an input vector with normally distributed noise. For duralava this vector is of length 64. Based on this random noise vector it generates a lava lamp image. The random vector thus encodes the state of the lava lamp. 

For training, the GAN is presented a real image of a lava lamp and also one of the fake lava lamp and then it learns to make the fake ones look as real as possible. 

For a lava lamp, a sequence of images has to be created. This sequence should in fact be infinite since a lava lamp can run forever. Thus the GAN should learn to output an arbitrarily long sequence of lava lamp images as a video. This is achieved by using a recurrent neural network (RNN). The RNN gets the 64 element noise vector of time step *t* and outputs the 64 element noise vector for time stemp *t+1*. 

The tricky part is to make sure that the state of the lava lamp (the 64 element random noise vector) remains stable. It could for example happen that over time the distribution of noise in the vector diverges from a normal distribution the mean becomes 10 and the standard deviation 52. In this case, the output images of the lava lamps wouldn't be correct anymore as the GAN was trained to expect the input vector to be normally distributed. To solve this problem, I make sure that in training the output of the RNN stays normally distributed. This is accomplished by adding penalization terms in the training which discourage the noise to diverge from the normal distribution. 

