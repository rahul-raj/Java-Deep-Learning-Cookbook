import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.ROC;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;

//Download dataset from https://skymindacademy.blob.core.windows.net/physionet2012/physionet2012.tar.gz
public class LstmTimeSeriesExample {
    private static final String FEATURE_DIR = "C:\\Users\\Admin\\Downloads\\physionet2012\\physionet2012\\sequence";
    private static final String LABEL_DIR = "C:\\Users\\Admin\\Downloads\\physionet2012\\physionet2012\\mortality";
    private static final int RANDOM_SEED = 1234;
    public static void main(String[] args) throws IOException, InterruptedException {
        SequenceRecordReader trainFeaturesReader = new CSVSequenceRecordReader(1, ",");
        trainFeaturesReader.initialize(new NumberedFileInputSplit(FEATURE_DIR+"/%d.csv",0,3199));
        SequenceRecordReader trainLabelsReader = new CSVSequenceRecordReader();
        trainLabelsReader.initialize(new NumberedFileInputSplit(LABEL_DIR+"/%d.csv",0,3199));
        DataSetIterator trainDataSetIterator = new SequenceRecordReaderDataSetIterator(trainFeaturesReader,trainLabelsReader,100,2,false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        SequenceRecordReader testFeaturesReader = new CSVSequenceRecordReader(1, ",");
        testFeaturesReader.initialize(new NumberedFileInputSplit(FEATURE_DIR+"/%d.csv",3200,3999));
        SequenceRecordReader testLabelsReader = new CSVSequenceRecordReader();
        testLabelsReader.initialize(new NumberedFileInputSplit(LABEL_DIR+"/%d.csv",3200,3999));
        DataSetIterator testDataSetIterator = new SequenceRecordReaderDataSetIterator(testFeaturesReader,testLabelsReader,100,2,false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
                                                        .seed(RANDOM_SEED)
                                                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                                                        .weightInit(WeightInit.XAVIER)
                                                        .updater(new Adam())
                                                        .dropOut(0.9)
                                                        .graphBuilder()
                                                        .addInputs("trainFeatures")
                                                        .setOutputs("predictMortality")
                                                        .addLayer("L1", new LSTM.Builder()
                                                                                       .nIn(86)
                                                                                        .nOut(200)
                                                                                        .forgetGateBiasInit(1)
                                                                                        .activation(Activation.TANH)
                                                                                        .build(),"trainFeatures")
                                                        .addLayer("predictMortality", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                                                            .activation(Activation.SOFTMAX)
                                                                                            .nIn(200).nOut(2).build(),"L1")
                                                        .build();

        ComputationGraph model = new ComputationGraph(configuration);

        for(int i=0;i<1;i++){
           model.fit(trainDataSetIterator);
           trainDataSetIterator.reset();
        }
        ROC evaluation = new ROC(100);
        while (testDataSetIterator.hasNext()) {
            DataSet batch = testDataSetIterator.next();
            INDArray[] output = model.output(batch.getFeatures());
            evaluation.evalTimeSeries(batch.getLabels(), output[0]);
        }
        
        System.out.println(evaluation.calculateAUC());
        System.out.println(evaluation.stats());
    }
}
