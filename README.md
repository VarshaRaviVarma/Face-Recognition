# Face-Recognition

## Abstract

A face recognition system is one of the biometric information processes, its applicability is easier and working range is larger than others, i.e.; fingerprint, iris scanning, signature, etc.Face detection and recognition is challenging due to the Wide variety of faces and the complexity of noises and image backgrounds. In this document, we propose a neural network based novel method for face recognition in cluttered and noisy images. 

This document is prepared by us to give a brief understanding on the work process of the project. The system uses a sinle technique: face recognition. The face detection is performed on live acquired images without any application field in mind. Processes utilized in the system are white balance correction, skin like region segmentation, facial feature extraction and face image extraction on a face candidate. Then a face classification method that uses FeedForward Neural Network is integrated in the system. The system is tested with a database generated in the laboratory with 30 people. The tested system has acceptable performance to recognize faces within intended limits.

## Introduction

### What is face recognition?

Face Recognition is one of the key areas under research. It has number of applications and uses. Many methods and algorithms are put forward like, 3D facial recognition etc. Face recognition comes under Biometric identification like iris, retina, finger prints etc. The features of the face are called biometric identifiers. The biometric identifiers are not easily forged, misplaced or shared hence access through biometric identifier gives us a better secure way to provide service and security.We can also develop many intelligent applications which may provide security and identity. These systems can be well incorporated into mobile and embedded systems efficiently and can be utilized on larger scale.

### Problem background

This current era is known for its modern science and technologies. Thus, we decided to take the latest advancements into consideration and work on a project based Artificial Intelligence. 

Face Recognition is one of the latest technologies being used widely. There are a lot of ways you could implement AI system such as face recognition. Here, implementing is not the hard part but building a model that can accurately do it is the hard part.

Thus, our project involves building a code for face detection and recognition using Artificial Intelligence (AI). In other words, the objective of our project is to design model that not only detects but also recognises faces from an image/video as accurately as possible.

### Problem description

The main problem in this project is to be able to recognise the face as accurately as possible (as mentioned in detail above). To be able to overcome this problem, we need to overcome few constraints related to Face Recognition mentioned below:

Image quality: Image quality affects how well facial recognition algorithms work. The image quality of scanning video is quite low compared with that of a digital camera.
Image size: When a face-detection algorithm finds a face in an image or in a still from a video capture, the relative size of that face compared with the enrolled image size affects how well the face will be recognized. 
Face angle: The relative angle of the target’s face influences the recognition score profoundly.
Processing and storage: Even though high-definition video is quite low in resolution when compared with digital camera images, it still occupies significant amounts of disk space.
Exposure: The system don’t recognize properly in poor light so may give false results.
Distance: It can only detect face from a limited distance.

### Significance 

This project can be implemented in various fields such as smartphones, laptops, personal computers, etc for further modified use of face recognition activities such as face unlock, tags, image organisation, etc.
Furthermore, it can be used for situations like attendance system, fraud analysis, image processing, security, etc.

## Literature Review

Zoltan Szlavik and Tamas Sziranyi [1]: Novel on-chip algorithms are proposed for face analysis on pictures obtained from a multi-camera surveillance system. According to the objective of the face detection sub-system, spatial positional relations to be presented for which the methodology we used is detailed in the paper. The uniqueness of the approach is the methodology we applied provides superior performance (50 fps) on standard database compared to most of the known face detection systems. In the general model based face part identification schema we relied on, the heavy use of morphologic operations gave robust results, even in analog VLSI implementation.

Geoffrey E. Hilton [2]: We trained a large, deep convolutional neural network to classify the 1.3 million high-resolution images in the LSVRC-2010 ImageNet training set into the 1000 different classes. On the test data, we achieved top-1 and top-5 error rates of 39.7\% and 18.9\% which is considerably better than the previous state-of-the-art results. The neural network, which has 60 million parameters and 500,000 neurons, consists of five convolutional layers, some of which are followed by max-pooling layers, and two globally connected layers with a final 1000-way softmax. To make training faster, we used non-saturating neurons and a very efficient GPU implementation of convolutional nets. To reduce overfitting in the globally connected layers we employed a new regularization method that proved to be very effective.

Kresimir Delac, Mislav Grgic and Marian Stewart Bartlett [3]: Face recognition is still a vividly researched area in computer science. First attempts were made in early 1970-ies, but a real boom happened around 1988, parallel with a large increase in computational power. The first widely accepted algorithm of that time was the PCA or eigenfaces method, which even today is used not only as a benchmark method to compare new methods to, but as a base for many methods derived from the original idea.

Alex Krizhevsky [4]: When training a convolutional DBN, one must decide what to do with the edge pixels of teh images. As the pixels near the edge of an image contribute to the fewest convolutional lter outputs, the model may see it t to tailor its few convolutional lters to better model the edge pixels. This is undesirable becaue it usually comes at the expense of a good model for the interior parts of the image. We investigate several ways of dealing with the edge pixels when training a convolutional DBN. Using a combination of locally-connected convolutional units and globally-connected units, as well as a few tricks to reduce the eects of overtting, we achieve state-of-the-art performance in the classication task of the CIFAR-10 subset of the tiny images dataset

Dr. Priya Guptaa , Nidhi Saxena a , Meetika Sharma a , Jagriti Tripathia [5]: Face recognition (FR), the process of identifying people through facial images, has numerous practical applications in the area of biometrics, information security, access control, law enforcement, smart cards and surveillance system. Convolutional Neural Networks (CovNets), a type of deep networks has been proved to be successful for FR. For real-time systems, some preprocessing steps like sampling needs to be done before using to CovNets. But then also complete images (all the pixel values) are passed as input to CovNets and all the steps (feature selection, feature extraction, training) are performed by the network. This is the reason that implementing CovNets are sometimes complex and time consuming. CovNets are at the nascent stage and the accuracies obtained are very high, so they have a long way to go. The paper proposes a new way of using a deep neural network (another type of deep network) for face recognition. In this approach, instead of providing raw pixel values as input, only the extracted facial features are provided. This lowers the complexity of while providing the accuracy of 97.05% on Yale faces dataset.

## System Design and Specifications

The following problem scope for this project was arrived at after reviewing the literature on face detection and face recognition, and determining possible real-world situations where such systems would be of use. The following system(s) requirements were identified 
1. A system to detect frontal view faces in static images 
2. A system to recognize a given frontal view face 
3. Only expressionless, frontal view faces will be presented to the face detection&recognition 
4. All implemented systems must display a high degree of lighting invariency. 
5. All systems must posses near real-time performance. 
6. Both fully automated and manual face detection must be supported 
7. Frontal view face recognition will be realised using only a single known image 
8. Automated face detection and recognition systems should be combined into a fully automated face detection and recognition system. The face recognition sub-system must display a slight degree of invariency to scaling and rotation errors in the segmented image extracted by the face detection sub-system. 
9. The frontal view face recognition system should be extended to a pose invariant face recognition system. Unfortunately although we may specify constricting conditions to our problem domain, it may not be possible to strictly adhere to these conditions when implementing a system in the realworld. 

## Implementation

An algorithm of system is implemented on MATLAB software. Image Acquisition Toolbox, Image Processing Toolbox, and Neural Network Toolbox are used for algorithm development.

### Face Detection
First implementation of system is performed on detection of faces in acquired image. Therefore, face detection has started with skin like region segmentation.

Besides RGB gives the best result, colors of wall in the background of an image can be skin like color due to white balance value of camera. Unwanted skin like color regions can affect detection and distort face shape. This color problem can be eliminated by white balance correction of acquired image. To guarantee color correction both segmentations are performed on acquired and corrected image, then logical “and operation” is performed.

### Face Recognition

With addition of isosceles triangle approach, as described in the previous section, eyes and mouth are found and cropped face image. Then, database of face images can be generated.

A database is generated from 30 people with 30 samples image for each person. Database is created from face detection part. Pattern Recognition Tool in Neural Network Toolbox is used to generate and train neural network. The generated network consists of 2 layers with sigmoid transfer function. 2 layers are hidden layer and output layer. Output layer has 26 neurons and Hidden layer neuron numbers are 41.

### Face Recognition System

Finally, face detection and recognition parts are merged to implement face recognition system. System can also handle more than one faces in the acquired image. Code is generated on MATLAB environment.
Many experiments are performed on live acquired images. Face detection and recognition parts performed well. Skin segmentation both decrease computational time and search area for face. Experiments show that connection is established well between detection and recognition parts.

## Results and Discussions

The computational models, which were implemented in this project, were chosen after extensive research, and the successful testing results confirm that the choices made by the researcher were reliable. This system was tested under very robust conditions in this experimental study and it is envisaged that real-world performance will be far more accurate.

The fully automated frontal view face recognition system displayed virtually perfect accuracy and in the researcher's opinion further work need not be conducted in this area. 

The implemented fully automated face recognition system could be used for simple surveillance applications such as ATM user security, while the implemented automated recognition system is ideal of mug shot matching. Since controlled conditions are present when mug shots are gathered, the frontal view face recognition scheme should display recognition accuracy far better than the results, which were obtained in this study, which was conducted under adverse conditions. 

Furthermore, many of the test subjects did not present an expressionless, frontal view to the system. They would probably be more compliant when a 6'5'' policeman is taking their mug shot! In mug shot matching applications, perfect recognition accuracy or an exact match is not a requirement. If a face recognition system can reduce the number of images that a human operator has to search through for a match from 10000 to even a 100, it would be of incredible practical use in law enforcement. 

The automated vision systems implemented in this document did not even approach the performance, nor were they as robust as a human's innate face recognition system. However, they give an insight into what the future may hold in computer vision.

## Conclusion

Our main aim was to build a face recognition model that is simple, user understandable and more accurate. Using the functions developed for data splitting, data transforming, data augmenting and the functions for face detection, face recognition a face recognition model was built were able to achieve the goal. This model is capable of detecting face and recognize it through the data provided for training the model. The model is currently able to provide an accuracy of above 75% which is great.
 
Once trained the system responds to single face recognition queries in less than 0.5 seconds. The system completed a query stream of 900 test images in 3 seconds, taking a per query time slot of just 3 milliseconds.

The system was tested under very robust conditions in this experimental study and it is envisaged that real-world performance will be far more accurate. The face detection system displayed virtually perfect accuracy and in the researcher's opinion further work need not be conducted in this area.

This model was basically designed for automated attendance system in colleges and schools but the model can also be implemented in various fields such as security, payments which is currently in experimental state, criminal identification, health, marketing and retail etc.

## References

1. Haralick, R.M. and Shapiro, L.G.. (1992) Computer and Robot Vision, Volume I. Addison-Wesley 
2. Haxby, J.V., Ungerleider, L.G., Horwitz, B., Maisog, J.M., Rapoport, S.I., and Grady, C.L. (1996). Face encoding and recognition in the human brain. Proc. Nat. Acad. Sci. 93: 922 - 927. 
3. Heisele, B. and Poggio, T. (1999) Face Detection. Artificial Intelligence Laboratory. MIT. 
4. Jang., J., Sun, C., and Mizutani, E. (1997) Neuro-Fuzzy and Soft Computing. Prentice Hall. 
5. Johnson, R.A., and Wichern, D.W. (1992) Applied Multivariate Statistical Analysis. Prentice Hall. p356-395
