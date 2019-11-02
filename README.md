


![Java deep learning cookbook](https://user-images.githubusercontent.com/517415/58750097-89578880-84ab-11e9-8863-ba65a6677374.png) []()

# Java Deep Learning Cookbook

This is a code repository for the upcoming book "Java Deep Learning cookbook" sponsored by Packt Publishing. We use and promote **deeplearning4j** library for all use-cases in this book. 
Official **deeplearning4j** version targeted in this cookbook is **1.0.0-beta3**. For the same reason, some of the methods or approaches may deprecated in their newer versions. So, be sure to refer their latest API documentation. You may use newer versions that has bug fixes and new features. 

**Update**

ETA for the cookbook release is  November 8, 2019.


# Build

Each chapter will have separate source folder where all examples are stored for the particular chapter. For example, if you want to import the code for chapter 2, navigate to the chapter directory first and then import the directory **sourceCode/cookbook-app** in your IDE. You should also see pom.xml located there.

![cookbookworkspace](https://user-images.githubusercontent.com/517415/56918244-20ab7380-6adb-11e9-95b9-b27e3550d985.png)
  
   ## From Intellij IDE

 - Navigate to the sourceCode root directory.
 - Open as a Maven project and compile.


## From Command Line
    mvn clean install

If you face issues with Intellij being not able to detect dependencies or any workspace issues,
try running the below command:

   
    mvn idea:idea

Delete ***workspace.xml*** under ***.idea*** directory if problem persists.


# Table of Contents


 1. [Introduction to Deep Learning in Java](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/01_Introduction_to_Deep_Learning_in_Java)
 2. [Data Extraction, Transform and Loading](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/02_Data_Extraction_Transform_and_Loading)
 3. [Building Deep Neural Networks for Binary classification](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/03_Building_Deep_Neural_Networks_for_Binary_classification)
 4. [Building Convolutional Neural Networks](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/04_Building_Convolutional_Neural_Networks)
 5. [Implementing NLP](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/05_Implementing_NLP)
 6. [Constructing LTSM Network for time series](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/06_Constructing_LSTM_Network_for_time_series)
 7. [Constructing LTSM Neural network for sequence classification](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/07_Constructing_LSTM_Neural_network_for_sequence_classification)
 8. [Performing Anomaly detection on unsupervised data](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/08_Performing_Anomaly_detection_on_unsupervised_data)
 9. [Using RL4J for Reinforcement learning](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/09_Using_RL4J_for_Reinforcement_learning)
 10. [Developing applications in distributed environment](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/10_Developing_applications_in_distributed_environment)
 11. [Applying Transfer Learning to network models](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/11_Applying_Transfer_Learning_to_network_models)
 12. [Benchmarking and Neural Network Optimization](https://github.com/rahul-raj/Java-Deep-Learning-Cookbook/tree/master/12_Benchmarking_and_Neural_Network_Optimization)

