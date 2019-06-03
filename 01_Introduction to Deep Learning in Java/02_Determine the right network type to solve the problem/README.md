


Although there are multiple ways to perform a single task, here are few of the commonly used network architectures for the mentioned use-cases:

| Problem  | Core Architecture |
|--|--|
| Image Classification | CNN |
| Anomaly Detection | Autoencoder |
| Time Series classification | RNN/LSTM/Computation graph
| Prediction problems on sequence data | RNN/LSTM
| Recommender Systems | RL


Note that, the optimal architectural decision can vary upon the type of data dealt with and whether it is supervised/unsupervised. 

 - For prediction problems with simple CSV data (non-sequential), MLP(Multilayer perceptron) would be just enough and will give best results compared to other complex architectures.  MLP is nothing but the deep neural net with an input layer and output layer and multiple hidden layers in between these two layers. Hidden layers receive input from the input layer and output layers receive input from hidden layers. If there are multiple hidden layers, each layer will take inputs from preceding hidden layer.
 
- Time series data or anything that involves sequential data is going to need a RNN or LSTM. RNN is optimal to handle sequential data. If we want to track long term dependencies in the data, an LSTM might be the best option. LSTM is a variant from RNN with a memory unit which is capable to hold long term dependencies.
- Anomaly detection problems involve feature analysis of each and every sample. We dont need to have labels here. We basically try to encode the data and decode it back to see the outliers in the feature. An autoencoder will be perfect fit for this purpose and lot of better variants like VAE (Variational autoencoder) are possible to construct using DL4J.
- DL4J have its on subsidiary library for reinforcement learning called RL4J. Recommender systems use reinforcement learning algorithms to solve recommendation problems. We can also feed the data in image/video/text format to a feed forward network/CNN and then generate classified actions. That is to chose the policy upon given action.  
 
