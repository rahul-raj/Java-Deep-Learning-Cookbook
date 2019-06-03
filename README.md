

![Java deep learning cookbook](https://user-images.githubusercontent.com/517415/58750097-89578880-84ab-11e9-8863-ba65a6677374.png) []()

# Java Deep Learning Cookbook

This is a code repository for the upcoming book "Java Deep Learning cookbook" sponsored by Packt Publishing. We use and promote **deeplearning4j** library for all use-cases in this book.  Cookbook will be released on or before September 2019 and codebase will be updated as the book progresses. README sections will be updated accordingly.

# Build

Each chapter will have separate source folder where all examples are stored for the particular chapter. For example, if you want to import the code for chapter 2, navigate to the chapter directory first and then import the directory **sourceCode/cookbook-app** in your IDE. You should also see pom.xml located there.

![cookbookworkspace](https://user-images.githubusercontent.com/517415/56918244-20ab7380-6adb-11e9-95b9-b27e3550d985.png)
  
   ## From Intellij IDE

 - Navigate to the sourceCode root directory.
 - Open as a Maven project and compile.


## [](https://github.expedia.biz/Brand-Expedia/ews-booking-service/#from-command-line)From Command Line
    mvn clean install

If you face issues with Intellij being not able to detect dependencies or any workspace issues,
try running the below command:

   
    mvn idea:idea

Delete ***workspace.xml*** under ***.idea*** directory if problem persists.


# Table of Contents


 1. [Introduction to Deep Learning in Java](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook#chapter-1--introduction-to-deep-learning-in-java)
 2. [Data Extraction, Transform and Loading](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook#chapter-2--data-extraction-transform-and-loading)
 3. [Building Deep Neural Networks for Binary classification](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook#chapter-3--building-deep-neural-networks-for-binary-classification)
 4. [Building Convolutional Neural Networks](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook#chapter-4--building-convolutional-neural-networks)
 5. [Implementing NLP](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook#chapter-5--implementing-nlp)
 6. [Constructing LTSM Network for time series](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook#chapter-6--constructing-lstm-network-for-time-series)
 7. [Constructing LTSM Neural network for sequence classification](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook#chapter-7--constructing-lstm-neural-network-for-sequence-classification)
 8. [Performing Anomaly detection on unsupervised data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook#chapter-8--performing-anomaly-detection-on-unsupervised-data)
 9. [Using RL4J for Reinforcement learning](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook#chapter-9--using-rl4j-for-reinforcement-learning)
 10. [Developing applications in distributed environment](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook#chapter-10--developing-applications-in-distributed-environment)
 11. [Applying Transfer Learning to network models](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook#chapter-11--applying-transfer-learning-to-network-models)
 12. [Benchmarking and Neural Network Optimization](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook#chapter-12--benchmarking-and-neural-network-optimization-in-dl4j)

## Chapter 1 : Introduction to Deep Learning in Java
In this chapter, we will discuss about DL4J as a distinct deep learning solution and the significance of Java deep learning library. We will also showcase required deep learning concepts in a recipe-based approach.

   1. **[Determine the right deep learning library](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/01_Introduction%20to%20Deep%20Learning%20in%20Java/01_Determine%20the%20right%20deep%20learning%20library)**.
   
         Choose the right DL library for your use-case.
       
   2. **[Determine the right network type to solve the problem](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/01_Introduction%20to%20Deep%20Learning%20in%20Java/02_Determine%20the%20right%20network%20type%20to%20solve%20the%20problem)**.
    
         Choose the right neural network type for your use-case.
       
   3. **[Determine the right activation function](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/01_Introduction%20to%20Deep%20Learning%20in%20Java/03_Determine%20the%20right%20activation%20function)**.
    
       Choose the right activation function for the network layers.
       
   4. **[Combat overfitting problems](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/01_Introduction%20to%20Deep%20Learning%20in%20Java/04_Combat%20overfitting%20problems)**.
    
       Here we learn how to overcome overfitting problems by a margin.
       
   5. **[Determine the right batch size and learning rates](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/01_Introduction%20to%20Deep%20Learning%20in%20Java/05_Determine%20the%20right%20batch%20size%20and%20learning%20rates)**.
    
        Determine the right batch size and learning rates to train the network.
        
   6. **[Configuring Maven  for  DL4J](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/01_Introduction%20to%20Deep%20Learning%20in%20Java/06_Configuring%20Maven%20%20for%20%20DL4J)**.
    
        Understand the required maven dependencies to run deeplearning4j in your own projects.
        
   7. **[Configuring DL4J for GPU accelerated environment](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/01_Introduction%20to%20Deep%20Learning%20in%20Java/07_Configuring%20DL4J%20for%20GPU%20accelerated%20environment)**.
    
        Learn how to configure DL4J to run on GPU powered machines.
        
   8. **[Troubleshooting Installation issues](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/01_Introduction%20to%20Deep%20Learning%20in%20Java/08_Troubleshooting%20Installation%20issues)**.
    
        Discuss most common issues encountered during DL4J installation.


## Chapter 2 : Data Extraction, Transform and Loading
   In this chapter, we will discuss about how to perform ETL(Extract, Transform & Loading) operations on redundant data before pass on to the deep neural networks. We will also talk about how to perform different data wrangling approaches using DL4J.

1. **[Read and Iterate through data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/02_Data%20Extraction,%20Transform%20and%20Loading/01_Read%20and%20Iterate%20through%20data)**

      Learn how to extract and process data using DataVec.
      
2. **[Perform Schema Transformation](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/02_Data%20Extraction,%20Transform%20and%20Loading/02_Perform%20Schema%20Transformation)**

      Learn how to transform data using DataVec.
      
3. **[Serializing transforms](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/02_Data%20Extraction,%20Transform%20and%20Loading/03_Serializing%20transforms)**

      Serialize transformed data.
      
4. **[Building transform process](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/02_Data%20Extraction,%20Transform%20and%20Loading/04_Building%20transform%20process)**

      Build transformation from defined schema.
      
5. **[Executing a transform process](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/02_Data%20Extraction,%20Transform%20and%20Loading/05_Executing%20a%20transform%20process)**

      Transform the data to required form.
      
6. **[Normalize data for network efficiency](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/02_Data%20Extraction,%20Transform%20and%20Loading/06_Normalize%20data%20for%20network%20efficiency)**

      Learn how to normalize data using DataVec.

## Chapter 3 : Building Deep Neural Networks for Binary Classification
In this chapter,  we will discuss on how to build deep neural networks for predicting customer retention in an organization. We will also follow step-by-step approach to solve the problem. The focus will be on how to make use of deep learning basics to create a deep neural network for binary classification from scratch.

1. **[Extracting data from CSV input](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/03_Building%20Deep%20Neural%20Networks%20for%20Binary%20classification/01_Extracting%20data%20from%20CSV%20input)**

    Perform data extraction from CSV file.

2. **[Removing anomalies from data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/03_Building%20Deep%20Neural%20Networks%20for%20Binary%20classification/02_Removing%20anomalies%20from%20data)**

    Remove redundant data or noises from the data.

3. **[Applying transformation to the data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/03_Building%20Deep%20Neural%20Networks%20for%20Binary%20classification/03_Applying%20transformation%20to%20the%20data)**

   Transform the data to required numerical format to use with neural network.

4. **[Design input layers for NN model](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/03_Building%20Deep%20Neural%20Networks%20for%20Binary%20classification/04_Design%20input%20layers%20for%20NN%20model)**

   Design input layers for multi-layered NN model.

5. **[Design hidden layers for NN model](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/03_Building%20Deep%20Neural%20Networks%20for%20Binary%20classification/05_Design%20hidden%20layers%20for%20NN%20model)**

   Design hidden layers for multi-layered NN model.

6. **[Design output layers for NN model](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/03_Building%20Deep%20Neural%20Networks%20for%20Binary%20classification/06_Design%20output%20layers%20for%20NN%20model)**

   Design output layers for multi-layered NN model.

7. **[Train and evaluate NN model for CSV data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/03_Building%20Deep%20Neural%20Networks%20for%20Binary%20classification/07_Train%20and%20evaluate%20%20NN%20model%20for%20CSV%20data)**

   Train the neural network for the CSV data and evaluate model accuracy.

8. **[Deploy NN model and use as an API](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/03_Building%20Deep%20Neural%20Networks%20for%20Binary%20classification/08_Deploy%20NN%20model%20and%20use%20as%20an%20API)**

   Deploy the neural network model into a web application and use as an API for your microservice.

## Chapter 4 : Building Convolutional Neural networks 
In this chapter,  we will discuss on how to build a convolutional neural network in DL4J for animal classification problem. We will also follow step-by-step approach to solve the problem. The focus will be on how to make use of CNN basics to construct an image recognition system from scratch.

1. **[Extract images from the disk](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/04_Building%20Convolutional%20Neural%20Networks/01_Extract%20images%20from%20the%20disk)**

   Retrieve image inputs from disk.

2. **[Create image variations for train data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/04_Building%20Convolutional%20Neural%20Networks/02_Create%20image%20variations%20for%20train%20data)**

   Create copies of images by flipping and rotating them.

3. **[Create image variations for train data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/04_Building%20Convolutional%20Neural%20Networks/03_Image%20pre-processing%20and%20design%20of%20input%20layers)**

   Perform sampling of image and design respective input layers.

4. **[Constructing hidden layers for CNN](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/04_Building%20Convolutional%20Neural%20Networks/04_Constructing%20hidden%20layers%20for%20CNN)**

   Construct hidden layers for the CNN.

5. **[Constructing output layers for output classification](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/04_Building%20Convolutional%20Neural%20Networks/05_Constructing%20output%20layers%20for%20output%20classification.)**

   Construct output layers for classified output.

6. **[Training images and evaluating CNN output](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/04_Building%20Convolutional%20Neural%20Networks/06_Training%20images%20and%20evaluating%20CNN%20output)**

   Train on image train data and evaluate the model performance.

7. **[API Deployment options](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/04_Building%20Convolutional%20Neural%20Networks/07_API%20Deployment%20options)**

   Utilize CNN model as API endpoint.

## Chapter 5 : Implementing NLP
In this chapter, we will discuss on paragraph vectors (doc2vec) and word vectors (word2vec) for text classification in DL4J. We will have a close look on how to create numeric vectors out of text. 
1. **[Read and load text data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/05_Implementing%20NLP/01_Read%20and%20load%20text%20data)**

   Load train data and configure word2vec models.

2. **[Tokenize data and Train the model](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/05_Implementing%20NLP/02_Tokenize%20data%20and%20Train%20the%20model)**

   Tokenize word vectors and train the model.

3. **[Evaluate the model](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/05_Implementing%20NLP/03_Evaluate%20the%20model)**

   Evaluate the resultant mode for the efficiency.

4. **[Visualize the model](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/05_Implementing%20NLP/04_Visualize%20the%20model)**

   Visualize the model using TSNE.

5. **[Save and reload model](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/05_Implementing%20NLP/05_Save%20and%20reload%20model)**

    Persist and reload the model from disk.

6. **[Import Google news vectors](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/05_Implementing%20NLP/06_Import%20Google%20news%20vectors)**

   Import other word2vec models such as Google news vectors.

7. **[Troubleshooting and Tuning Word2Vec models](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/05_Implementing%20NLP/07_Troubleshooting%20and%20Tuning%20Word2Vec%20models)**

   Performance tuning for word2vec models.

8. **[Using word2vec for sentence classification using convolutional neural network](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/05_Implementing%20NLP/08_Using%20%20word2vec%20for%20sentence%20classification%20using%20convolutional%20neural%20network)**

   Learn how to use Word2Vec for convolutional neural network.

9. **[Using Doc2Vec for document classification](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/05_Implementing%20NLP/09_Using%20%20Doc2Vec%20for%20document%20classification)**

   Learn different use cases where Where we can leverage Doc2Vec

## Chapter 6 : Constructing LSTM network for time series

In this chapter,  we will solve a clinical time series problem to perform patient mortality based on a number of factors. We will be developing LSTM(Long Short-Term Memory) network for medical time series problem. A step-by-step approach will be followed throughout the chapter. The focus will be on how to apply what you have learned on recurrent neural networks and time series to solve a real-world challenge in the medical field.

1. **[Extract and read clinical data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/06_Constructing%20LTSM%20Network%20for%20time%20series/01_Extract%20and%20read%20clinical%20data)**

   Download and prepare the clinical data.

2. **[Load and transform data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/06_Constructing%20LTSM%20Network%20for%20time%20series/02_Load%20and%20transform%20data)**

   Convert raw data to ready to train objects.

3. **[Constructing input layers for the network](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/06_Constructing%20LTSM%20Network%20for%20time%20series/03_Constructing%20input%20layers%20for%20the%20%20network)**

   Build input layers for the mortality test.

4. **[Constructing output layers for the network](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/06_Constructing%20LTSM%20Network%20for%20time%20series/04_Constructing%20output%20layers%20for%20the%20network)**

   Build output layers for the mortality test.

5. **[Train time series data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/06_Constructing%20LTSM%20Network%20for%20time%20series/05_Train%20time%20series%20data)**

   Train the clinical time series data.

6. **[Evaluate the LSTM network efficiency.](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/06_Constructing%20LTSM%20Network%20for%20time%20series/06_Evaluate%20the%20LSTM%20network%20efficiency)**

   Evaluate the network results.

## Chapter 7 : Constructing LSTM Neural network for sequence classification
In this chapter,  we will discuss on how to construct LSTM recurrent neural network for time series sequence classification. We will also follow step-by-step approach throughout the chapter. The focus will be on how to apply what you have learned on recurrent neural networks and time series to solve sequence classification problem.

1. **[Extract time series data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/07_Constructing%20LTSM%20Neural%20network%20for%20sequence%20classification/01_Extract%20%20time%20series%20data)**

   Download Synthetic Control Chart Time Series Data.

2. **[Load the training data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/07_Constructing%20LTSM%20Neural%20network%20for%20sequence%20classification/02_Load%20the%20training%20data)**

   Load and prepare time series data to prior to training.

3. **[Normalize training data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/07_Constructing%20LTSM%20Neural%20network%20for%20sequence%20classification/03_Normalize%20training%20data)**

   Normalize time series data before passing to LSTM network.

4. **[Constructing input layers for the network](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/07_Constructing%20LTSM%20Neural%20network%20for%20sequence%20classification/04_Constructing%20input%20layers%20for%20the%20network)**

   Build input layers for the time series problem.

5. **[Constructing output layers for the network](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/07_Constructing%20LTSM%20Neural%20network%20for%20sequence%20classification/05_Constructing%20output%20layers%20for%20the%20network)**

   Build output layers for the time series problem.

6. **[Evaluate LSTM network for classified output.](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/07_Constructing%20LTSM%20Neural%20network%20for%20sequence%20classification/06_Evaluate%20LSTM%20network%20for%20classified%20output)**

   Evaluate sequence classifier outputs.

## Chapter 8 : Performing Anomaly detection on unsupervised data
In this chapter,  we will implement an autoencoder for MNIST anomaly detection. We will also follow step-by-step approach throughout the chapter. The focus will be on how to perform anomaly detection on unsupervised data by designing an autoencoder for the task. 

1. **[Extract and prepare MNIST data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/08_Performing%20Anomaly%20detection%20on%20unsupervised%20data/01_Extract%20%20and%20prepare%20MNIST%20data).**

      Extract and prepare required MNIST data.

2. **[Constructing LSTM layers for input](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/08_Performing%20Anomaly%20detection%20on%20unsupervised%20data/02_Constructing%20LSTM%20layers%20for%20input)**

      Build LSTM layers for the auto encoder.

3. **[Constructing output layers](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/08_Performing%20Anomaly%20detection%20on%20unsupervised%20data/03_Constructing%20output%20layers)**

   Build output layers for the auto encoder.

4. **[Train with MNIST images.](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/08_Performing%20Anomaly%20detection%20on%20unsupervised%20data/04_Train%20with%20MNIST%20images)**

   Train encoder with MNIST images.

5. **[Evaluate and sort the results based on anomaly score.](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/08_Performing%20Anomaly%20detection%20on%20unsupervised%20data/05_Evaluate%20and%20sort%20the%20results%20based%20on%20anomaly%20score)**

   Evaluate the anomaly score

6. **[Save the resultant model.](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/08_Performing%20Anomaly%20detection%20on%20unsupervised%20data/06_Save%20the%20resultant%20model)**

   Save the resultant model for reuse.

## Chapter 9 : Using RL4J for Reinforcement learning

In this chapter, we will discuss on how to develop more advanced neural networks for complex challenges. We will learn various implementations of reinforcement learning in RL4J.

1. **[Setting up environment and dependencies](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/09_Using%20RL4J%20for%20Reinforcement%20learning/01_Setting%20up%20environment%20and%20dependencies)**

   Learn about required dependencies for RL4J.

2. **[Setting up data requirements](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/09_Using%20RL4J%20for%20Reinforcement%20learning/02_Setting%20up%20data%20requirements)**

   Learn the type of data being used and to handle using RL4J.

3. **[Constructing Action-Reward feedback loop for the agent](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/09_Using%20RL4J%20for%20Reinforcement%20learning/03_Constructing%20Action-Reward%20feedback%20loop%20for%20the%20agent)**

   Construct the action-reward feedback system.

4. **[Visualize training and monitor agent progress using webapp-rl4j](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/09_Using%20RL4J%20for%20Reinforcement%20learning/04_Visualize%20training%20and%20monitor%20agent%E2%80%99s%20progress%20using%20webapp-rl4j)**

   Visualize the agent’s progress and training.


## Chapter 10 : Developing applications in distributed environment

In this chapter, we will discuss on how to construct neural networks for a distributed environment using DL4J and Spark. We will also discuss in detail about various performance optimization that can be applied for distributed training and evaluation.

1. **[Setup DL4J and required dependencies](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/10_Developing%20applications%20in%20distributed%20environment/01_Setup%20DL4J%20and%20required%20dependencies)**

   Setup DL4J for spark and add required maven dependencies.

2. **[Creating uber-Jar for training](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/10_Developing%20applications%20in%20distributed%20environment/02_Creating%20uber-Jar%20for%20training)**

   Create maven config to create a uber-Jar to spark submit.

3. **[Configuring to use CPU/GPU for training](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/10_Developing%20applications%20in%20distributed%20environment/03_Configuring%20to%20use%20CPU-GPU%20for%20training)**

   Configure CPU/GPU for training on distributed environment.

4. **[Memory settings and Garbage collection for Spark.](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/10_Developing%20applications%20in%20distributed%20environment/04_Memory%20settings%20and%20Garbage%20collection%20for%20Spark)**

   Apply performance optimization techniques.

5. **[Configuring Encoding thresholds](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/10_Developing%20applications%20in%20distributed%20environment/05_Configuring%20Encoding%20thresholds)**

   Optimize encoding threshold value for performance.

6. **[Performing Distributed Test set Evaluation](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/10_Developing%20applications%20in%20distributed%20environment/06_Performing%20Distributed%20Test%20set%20Evaluation)**

   Execute neural network evaluation while effort is distributed among clusters.

7. **[Saving and loading trained neural network models](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/10_Developing%20applications%20in%20distributed%20environment/07_Saving%20and%20loading%20trained%20neural%20network%20models)**

   Save/Load neural network model after training.

8. **[Performing Distributed Evaluation](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/10_Developing%20applications%20in%20distributed%20environment/08_Performing%20Distributed%20Evaluation)**

   Make distributed inference for inputs.

## Chapter 11 : Applying Transfer Learning to network models
In this chapter, we will discuss on DL4J transfer learning API functionalities. We will have a close look on how to modify or optimize model configuration in a pre-trained model from DL4J zoo. We will also talk about how to import models from Keras and use them in your DL4J code.

1. **[Modify an existing customer retention model](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/11_Applying%20Transfer%20Learning%20to%20network%20models/01_Modify%20an%20existing%20customer%20retention%20model)**

   Modify pre-trained DL4J model.

2. **[Fine-tune learning configuration](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/11_Applying%20Transfer%20Learning%20to%20network%20models/02_Fine-tune%20learning%20configuration)**

   Fine tune learning configuration in pre-trained model

3. **[Implementing frozen layers](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/11_Applying%20Transfer%20Learning%20to%20network%20models/03_Implementing%20frozen%20layers)**

   Add frozen layers to a pre-trained model.

4. **[Import and load Keras models and layers](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/11_Applying%20Transfer%20Learning%20to%20network%20models/04_Import%20and%20load%20Keras%20models%20and%20layers)**

   Import network models and layers  from Keras.

## Chapter 12 : Benchmarking and Neural Network Optimization in DL4J
In this chapter, we will discuss on DL4J benchmarking principles and various ways to optimize neural network performance in DL4J. We will be discussing performance optimization strategies such as memory management, garbage collection and hyperparameter optimization using Arbiter.

1. **[General Guidelines](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/12_Benchmarking%20and%20Neural%20Network%20Optimization/01_General%20Guidelines)**

   Learn general benchmarking principles for performance.

2. **[DL4J/ND4J specific configuration](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/12_Benchmarking%20and%20Neural%20Network%20Optimization/02_DL4J-ND4J%20specific%20configuration)**

   Configure DL4J/ND4J specific benchmarks.

3. **[Setting up heap space and garbage collection.](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/12_Benchmarking%20and%20Neural%20Network%20Optimization/03_Setting%20up%20heap%20space%20and%20garbage%20collection)**

   Configure optimal heap space and memory management. 

4. **[Using Asynchronous ETL](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/12_Benchmarking%20and%20Neural%20Network%20Optimization/04_Using%20Asynchronous%20ETL)**

   Configure Async ETL for improving ETL workload time.

5. **[Using Arbiter to monitor neural network behavior.](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/12_Benchmarking%20and%20Neural%20Network%20Optimization/05_Using%20Arbiter%20to%20monitor%20neural%20network%20behavior)**

   Explore Arbiter to observe neural network behavior.

6. **[Perform hyperparameter tuning](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/12_Benchmarking%20and%20Neural%20Network%20Optimization/06_Perform%20hyperparameter%20tuning)**

   Run a hyperparameter tuning to yield best network config.
