# Diabetic-Retinopathy-and-Predicting-its-Progression.

## Abstract

Diabetic retinopathy is a rising complication of diabetes that can lead to vision loss when it is not properly treated. Several studies have been extensively conducted on classifying the severity grades and segmenting retinopathy lesions from retinal images. However, there are few studies that focus on the longitudinal dynamics of the disease progression. This work focuses on a longitudinal study of diabetic retinopathy using fundus photographs. Two models were employed: autoencoders for image compression to provide a latent representation, and neural ordinary differential equations to predict the dynamics. The autoencoder model was trained to find the best dimension of the latent space that is relevant for predicting diabetic retinopathy grade. In the latent representation, the neural
ordinary differential equation model applied four solvers, where the Dormand-Prince5
solver achieved the best structural similarity of the true fundus photograph and the predicted one. The results show potential for predicting the advancement of retinopathy due to disease progression.

Keywords: diabetic retinopathy, longitudinal study, autoencoders, neural ordinary differential equations, artificial intelligence, deep learning, image processing

## Key questions
* Given a specific point of time in the future, what will patient's condition be like? what was changed in the mean time?
![What has changed in course of two years?](https://github.com/user-attachments/assets/cf61a7d6-cc19-4d66-939c-d9f10db6eb1a)

## Models 
Overview of the general proposed framework for Diabetic Retinopathy progression using NODE. The approach consists of two main tasks: (1) Learning efficient latent representations by fusing multimodal data (retinal images and clinical examination data) through reconstruction and classification objectives; (2) Modeling the temporal dynamics of DR progression using Neural Ordinary Differential Equations, enabling prediction of future DR grades and estimation of time-to-progression based on patient history.
![Detailed Pipeline of the Project](diagram.pdf)

Combined autoencoder and NODE model for predicting future states of diabetic
retinopathy. Color-codes show which true reconstruction corresponds to the latent representation with the same color. DN represents data.
![Training Processing of models](node.pdf)

Autoencoders architecture: conv2d is convolutional operation and conv2d.T is convolutional transpose. d is the number of latent channels. σ represents sigmoid activation function.
![Autoencoders architecture](AUTOENCODER.pdf)





# ABSTRACT
Diabetic retinopathy is a rising complication of diabetes that can lead to vision loss when it is not properly treated. Several studies have been extensively conducted on classifying the severity grades and segmenting retinopathy lesions from retinal images. However, there are few studies that focus on the longitudinal dynamics of the disease progression. This work focuses on a longitudinal study of diabetic retinopathy using fundus photographs. Two models were employed:autoencoders for image compression to provide a latent representation, and neural ordinary differential equations to predict the dynamics. The autoencoder model was trained to find the best dimension of the latent space that is relevant for predicting diabetic retinopathy grade. In the latent representation, the neural ordinary differential equation model applied four solvers, where the Dormand-Prince5 solver achieved the best structural similarity of the true fundus photograph and the predicted one. The results show potential for predicting the advancement of retinopathy due to disease progression.
