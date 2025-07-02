# Diabetic-Retinopathy-and-Predicting-its-Progression.

## Key questions
* Given a specific point of time in the future, what will patient's condition be like?
![diff_maps_animation_thresholded](https://github.com/user-attachments/assets/cf61a7d6-cc19-4d66-939c-d9f10db6eb1a)

# ABSTRACT
Diabetic retinopathy is a rising complication of diabetes that can lead to vision loss when it is not properly treated. Several studies have been extensively conducted on classifying the severity grades and segmenting retinopathy lesions from retinal images. However, there are few studies that focus on the longitudinal dynamics of the disease progression. This work focuses on a longitudinal study of diabetic retinopathy using fundus photographs. Two models were employed:autoencoders for image compression to provide a latent representation, and neural ordinary differential equations to predict the dynamics. The autoencoder model was trained to find the best dimension of the latent space that is relevant for predicting diabetic retinopathy grade. In the latent representation, the neural ordinary differential equation model applied four solvers, where the Dormand-Prince5 solver achieved the best structural similarity of the true fundus photograph and the predicted one. The results show potential for predicting the advancement of retinopathy due to disease progression.
