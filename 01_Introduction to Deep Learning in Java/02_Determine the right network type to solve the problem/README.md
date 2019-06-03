Although there are multiple ways to perform a single task, here are few of the commonly used network architectures for the mentioned use-cases:

| Problem  | Core Architecture |
|--|--|
| Image Classification | CNN |
| Anomaly Detection | Autoencoder |
| Time Series classification | RNN/LSTM/Computation graph
| Prediction problems on sequence data | RNN/LSTM
| Recommender Systems | RL


Note that, the optimal architectural decision can vary upon the type of data dealt with and whether it is supervised/unsupervised. 

For prediction problems with simple CSV data (non-sequential), MLP(Multilayer perceptron) would be just enough and will give best results compared to other complex architectures. 
