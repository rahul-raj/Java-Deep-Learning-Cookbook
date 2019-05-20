import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;

import java.io.IOException;
import java.util.*;

public class MnistAnomalyDetectionExampleCopy {
    public static void main(String[] args) throws IOException {

        DataSetIterator iterator = new MnistDataSetIterator(100,50000,false);
        int nEpochs = 30;
        //Random rng = new Random(12345);
        DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(iterator,500,0.8);

        MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                .seed(12345)
                .weightInit(WeightInit.XAVIER)
                .updater(new AdaGrad(0.05))
                .activation(Activation.RELU)
                .l2(.0001)
                .list()
                .layer(new DenseLayer.Builder().nIn(784).nOut(250)
                        .build())
                .layer(new DenseLayer.Builder().nIn(250).nOut(10)
                        .build())
                .layer(new DenseLayer.Builder().nIn(10).nOut(250)
                        .build())
                .layer(new OutputLayer.Builder().nIn(250).nOut(784)
                        .lossFunction(LossFunctions.LossFunction.MSE)
                        .build())
                .build();
        MultiLayerNetwork model = new MultiLayerNetwork(configuration);
        model.setListeners(new ScoreIterationListener(100));
        model.fit(splitter.getTrainIterator(),nEpochs);

        Map<Integer,List<Pair<Double,INDArray>>> listsByDigit = new HashMap<>();
        for( int i=0; i<10; i++ ){
            listsByDigit.put(i,new ArrayList<>());
        }

        DataSetIterator testIterator = splitter.getTestIterator();
        while(testIterator.hasNext()) {
            DataSet testSample = testIterator.next();
            for(int i=0;i<testSample.numExamples();i++) {
               DataSet data = testSample.get(i);
               INDArray features = data.getFeatures();
               INDArray labels = data.getLabels();
               Double score = model.score(new DataSet(features,labels));
            }


        }


    }
}