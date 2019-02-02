
# Java Deep Learning Cookbook



# Table of Contents


 1. Introduction to Deep Learning in Java
 2. Data Extraction, Transform and Loading
 3. Building Deep Neural Networks for Binary classification
 4. Building Convolutional Neural Networks
 5. Implementing NLP
 6. Constructing LTSM Network for time series
 7. Constructing LTSM Neural network for sequence classification
 8. Performing Anomaly detection on unsupervised data
 9. Using RL4J for Reinforcement learning
 10. Developing applications in distributed environment
 11. Applying Transfer Learning to network models
 12. Benchmarking and Neural Network Optimization

## Chapter 1 : Introduction to Deep Learning in Java
In this chapter, we will discuss about DL4J as a distinct deep learning solution and the significance of Java deep learning library. We will also showcase required deep learning concepts in a recipe-based approach.

   1. **Determine the right deep learning library**.
   
       Choose the right DL library for your use-case.
       
    2. **Determine the right network type to solve the problem**.
    
       Choose the right neural network type for your use-case.
       
    3. **Determine the right activation function**.
    
       Choose the right activation function for the network layers.
       
    4. **Combat overfitting problems**.
    
       Here we learn how to overcome overfitting problems by a margin.
       
    5. **Determine the right batch size and learning rates**.
    
        Determine the right batch size and learning rates to train the network.
        
    6. **Configuring Maven  for  DL4J**.
    
        Understand the required maven dependencies to run deeplearning4j in your own projects.
        
    7. **Configuring DL4J for GPU accelerated environment**.
    
        Learn how to configure DL4J to run on GPU powered machines.
        
    8. **Troubleshooting Installation issues**.
    
        Discuss most common issues encountered during DL4J installation.


## Chapter 2 : Data Extraction, Transform and Loading
   In this chapter, we will discuss about how to perform ETL(Extract, Transform & Loading) operations on redundant data before pass on to the deep neural networks. We will also talk about how to perform different data wrangling approaches using DL4J.

1. **Read and Iterate through data**

      Learn how to extract and process data using DataVec.
      
2. **Perform Schema Transformation**

      Learn how to transform data using DataVec.
      
3. **Serializing transforms**

      Serialize transformed data.
      
4. **Building transform process**

      Build transformation from defined schema.
      
5. **Executing a transform process**

      Transform the data to required form.
      
6. **Normalize data for network efficiency**

       Learn how to normalize data using DataVec.

## Chapter 3 : Building Deep Neural Networks for Binary Classification
In this chapter,  we will discuss on how to build deep neural networks for predicting customer retention in an organization. We will also follow step-by-step approach to solve the problem. The focus will be on how to make use of deep learning basics to create a deep neural network for binary classification from scratch.

1. **Extracting data from CSV input**

Perform data extraction from CSV file.

2. **Removing anomalies from data**

Remove redundant data or noises from the data.

3. **Applying transformation to the data**

Transform the data to required numerical format to use with neural network.

4. **Design input layers for NN model**

Design input layers for multi-layered NN model.

5. **Design hidden layers for NN model**

Design hidden layers for multi-layered NN model.

6. **Design output layers for NN model**

Design output layers for multi-layered NN model.

7. **Train and evaluate NN model for CSV data**

Train the neural network for the CSV data and evaluate model accuracy.

8. **Deploy NN model and use as an API**

Deploy the neural network model into a web application and use as an API for your microservice.

## Chapter 4 : Building Convolutional Neural networks 
In this chapter,  we will discuss on how to build a convolutional neural network in DL4J for animal classification problem. We will also follow step-by-step approach to solve the problem. The focus will be on how to make use of CNN basics to construct an image recognition system from scratch.

1. **Extract images from the disk**

Retrieve image inputs from disk.

2. **Create image variations for train data**

Create copies of images by flipping and rotating them.

3. **Create image variations for train data**

Perform sampling of image and design respective input layers.

4. **Constructing hidden layers for CNN**

Construct hidden layers for the CNN.

5. **Constructing output layers for output classification**

Construct output layers for classified output.

6. **Training images and evaluating CNN output**

Train on image train data and evaluate the model performance.

7. **API Deployment options**

Utilize CNN model as API endpoint.

## Chapter 5 : Implementing NLP
In this chapter, we will discuss on paragraph vectors (doc2vec) and word vectors (word2vec) for text classification in DL4J. We will have a close look on how to create numeric vectors out of text. 
1. **Read and load text data**

Load train data and configure word2vec models.

2. **Tokenize data and Train the model**

Tokenize word vectors and train the model.

3. **Evaluate the model**

Evaluate the resultant mode for the efficiency.

4. **Visualize the model**

Visualize the model using TSNE.

5. **Save and reload model**

Persist and reload the model from disk.

6. **Import Google news vectors**

Import other word2vec models such as Google news vectors.

7. **Troubleshooting and Tuning Word2Vec models**

Performance tuning for word2vec models.

8. **Using word2vec for sentence classification using convolutional neural network**

Learn how to use Word2Vec for convolutional neural network.

9. **Using Doc2Vec for document classification**

Learn different use cases where Where we can leverage Doc2Vec

## Chapter 6 : Constructing LSTM network for time series

In this chapter,  we will solve a clinical time series problem to perform patient mortality based on a number of factors. We will be developing LSTM(Long Short-Term Memory) network for medical time series problem. A step-by-step approach will be followed throughout the chapter. The focus will be on how to apply what you have learned on recurrent neural networks and time series to solve a real-world challenge in the medical field.

1. **Extract and read clinical data**

Download and prepare the clinical data.

2. **Load and transform data**

Convert raw data to ready to train objects.

3. **Constructing input layers for the network**

Build input layers for the mortality test.

4. **Constructing output layers for the network**

Build output layers for the mortality test.

5. **Train time series data**

Train the clinical time series data.

6. **Evaluate the LSTM network efficiency.**

Evaluate the network results.

## Chapter 7 : Constructing LSTM Neural network for sequence classification
In this chapter,  we will discuss on how to construct LSTM recurrent neural network for time series sequence classification. We will also follow step-by-step approach throughout the chapter. The focus will be on how to apply what you have learned on recurrent neural networks and time series to solve sequence classification problem.

1. **Extract time series data**

Download Synthetic Control Chart Time Series Data.

2. **Load the training data**

Load and prepare time series data to prior to training.

3. **Normalize training data**

Normalize time series data before passing to LSTM network.

4. **Constructing input layers for the network**

Build input layers for the time series problem.

5. **Constructing output layers for the network**

Build output layers for the time series problem.

6. **Evaluate LSTM network for classified output.**

Evaluate sequence classifier outputs.

## Chapter 8 : Performing Anomaly detection on unsupervised data
In this chapter,  we will implement an autoencoder for MNIST anomaly detection. We will also follow step-by-step approach throughout the chapter. The focus will be on how to perform anomaly detection on unsupervised data by designing an autoencoder for the task. 

1. **Extract and prepare MNIST data.**

Extract and prepare required MNIST data.

2. **Constructing LSTM layers for input**

Build LSTM layers for the auto encoder.

3. **Constructing output layers**

Build output layers for the auto encoder.

4. **Train with MNIST images.**

Train encoder with MNIST images.

5. **Evaluate and sort the results based on anomaly score.**

Evaluate the anomaly score

6. **Save the resultant model.**

Save the resultant model for reuse.

## Chapter 9 : Using RL4J for Reinforcement learning

In this chapter, we will discuss on how to develop more advanced neural networks for complex challenges. We will learn various implementations of reinforcement learning in RL4J.

1. **Setting up environment and dependencies**

Learn about required dependencies for RL4J.

2. **Setting up data requirements**

Learn the type of data being used and to handle using RL4J.

3. **Constructing Action-Reward feedback loop for the agent**

Construct the action-reward feedback system.

4. **Visualize training and monitor agent progress using webapp-rl4j**

Visualize the agent’s progress and training.


## Chapter 10 : Developing applications in distributed environment

In this chapter, we will discuss on how to construct neural networks for a distributed environment using DL4J and Spark. We will also discuss in detail about various performance optimization that can be applied for distributed training and evaluation.

1. **Setup DL4J and required dependencies**

Setup DL4J for spark and add required maven dependencies.

2. **Creating uber-Jar for training**

Create maven config to create a uber-Jar to spark submit.

3. **Configuring to use CPU/GPU for training**

Configure CPU/GPU for training on distributed environment.

4. **Memory settings and Garbage collection for Spark.**

Apply performance optimization techniques.

5. **Configuring Encoding thresholds**

Optimize encoding threshold value for performance.

6. **Performing Distributed Test set Evaluation**

Execute neural network evaluation while effort is distributed among clusters.

7. **Saving and loading trained neural network models**

Save/Load neural network model after training.

8. **Performing Distributed Evaluation**

Make distributed inference for inputs.

## Chapter 11 : Applying Transfer Learning to network models
In this chapter, we will discuss on DL4J transfer learning API functionalities. We will have a close look on how to modify or optimize model configuration in a pre-trained model from DL4J zoo. We will also talk about how to import models from Keras and use them in your DL4J code.

1. **Modify an existing customer retention model**

Modify pre-trained DL4J model.

2. **Fine-tune learning configuration**

Fine tune learning configuration in pre-trained model

3. **Implementing frozen layers**

Add frozen layers to a pre-trained model.

4. **Import and load Keras models and layers**

Import network models and layers  from Keras.

## Chapter 12 : Benchmarking and Neural Network Optimization in DL4J
In this chapter, we will discuss on DL4J benchmarking principles and various ways to optimize neural network performance in DL4J. We will be discussing performance optimization strategies such as memory management, garbage collection and hyperparameter optimization using Arbiter.

1. **General Guidelines**

Learn general benchmarking principles for performance.

2. **DL4J/ND4J specific configuration**

Configure DL4J/ND4J specific benchmarks.

3. **Setting up heap space and garbage collection.**

Configure optimal heap space and memory management. 

4. **Using Asynchronous ETL**

Configure Async ETL for improving ETL workload time.

5. **Using Arbiter to monitor neural network behavior.**

Explore Arbiter to observe neural network behavior.

6. **Perform hyperparameter tuning**

Run a hyperparameter tuning to yield best network config.
