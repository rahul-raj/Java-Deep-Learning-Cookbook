import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Nadam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class UciSequenceClassificationExample {

    //Replace with your file system location where you want to store feature/labels from data after extraction.
    static String trainfeatureDir = "D:/train/features/";
    static String trainlabelDir = "D:/train/labels/";
    static String testlabelDir = "D:/test/labels/";
    static String testfeatureDir = "D:/test/features/";


    private static Logger log = LoggerFactory.getLogger(UciSequenceClassificationExample.class);

    public static void main(String[] args) throws Exception {

        downloadUCIData();

        SequenceRecordReader trainFeaturesSequenceReader = new CSVSequenceRecordReader();
        trainFeaturesSequenceReader.initialize(new NumberedFileInputSplit(new File(trainfeatureDir).getAbsolutePath()+"/%d.csv",0,449));
        SequenceRecordReader trainLabelsSequenceReader = new CSVSequenceRecordReader();
        trainLabelsSequenceReader.initialize(new NumberedFileInputSplit(new File(trainlabelDir).getAbsolutePath()+"/%d.csv",0,449));

        SequenceRecordReader testFeaturesSequenceReader = new CSVSequenceRecordReader();
        testFeaturesSequenceReader.initialize(new NumberedFileInputSplit(new File(testfeatureDir).getAbsolutePath()+"/%d.csv",0,149));
        SequenceRecordReader testLabelsSequenceReader = new CSVSequenceRecordReader();
        testLabelsSequenceReader.initialize(new NumberedFileInputSplit(new File(testlabelDir).getAbsolutePath()+"/%d.csv",0,149));

        int batchSize = 10;
        int numOfClasses = 6;

        DataSetIterator trainIterator = new SequenceRecordReaderDataSetIterator(trainFeaturesSequenceReader,trainLabelsSequenceReader,batchSize,numOfClasses,false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
        DataSetIterator testIterator = new SequenceRecordReaderDataSetIterator(testFeaturesSequenceReader,testLabelsSequenceReader,batchSize,numOfClasses,false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        DataNormalization normalization = new NormalizerStandardize();
        normalization.fit(trainIterator);
        trainIterator.setPreProcessor(normalization);
        testIterator.setPreProcessor(normalization);

        ComputationGraphConfiguration configuration = new NeuralNetConfiguration.Builder()
                                                    .seed(123)
                                                    .weightInit(WeightInit.XAVIER)
                                                    .updater(new Nadam())
                                                    .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                                                    .gradientNormalizationThreshold(0.5)
                                                    .graphBuilder()
                                                    .addInputs("trainFeatures")
                                                    .setOutputs("predictSequence")
                                                    .addLayer("L1",new LSTM.Builder().activation(Activation.TANH).nIn(1).nOut(10).build(),"trainFeatures")
                                                    .addLayer("predictSequence",new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                                                                                                            .activation(Activation. SOFTMAX).nIn(10).nOut(numOfClasses).build(),"L1")
                                                    .build();
        ComputationGraph model = new ComputationGraph(configuration);
        model.init();
        log.info("Starting training...");

        //Print the score (loss function value) every 20 iterations
        model.setListeners(new ScoreIterationListener(20), new EvaluativeListener(testIterator, 1, InvocationType.EPOCH_END));

        int nEpochs = 100;
        model.fit(trainIterator,nEpochs);

        log.info("Evaluating...");
        Evaluation evaluation = model.evaluate(testIterator);
        System.out.println(evaluation.stats());

    }

    private static void downloadUCIData() throws Exception {
        final String url = "https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data";
        final String data = IOUtils.toString(new URL(url),"utf-8");
        final String[] sequences = data.split("\n");
        final List<Pair<String,Integer>> contentAndLabels = new ArrayList<>();
        int lineCount = 0;
        for(String sequence : sequences) {
            //Labels: first 100 examples (lines) are label 0, second 100 examples are label 1, and so on
            sequence = sequence.replaceAll(" +","\n");
            contentAndLabels.add(new Pair<>(sequence, lineCount++ / 100));
        }
        Collections.shuffle(contentAndLabels,new Random(12345));


        int trainCount=0;
        int testCount=0;
        final int traintestSplit = 450;
        File featureFile;
        File labelFile;
        for(Pair<String,Integer> sequencePair : contentAndLabels) {
            if(trainCount<traintestSplit) {
                featureFile = new File(trainfeatureDir+trainCount+".csv");
                labelFile = new File(trainlabelDir+trainCount+".csv");
                trainCount++;
            } else {
                featureFile = new File(testfeatureDir+testCount+".csv");
                labelFile = new File(testlabelDir+testCount+".csv");
                testCount++;
            }

            FileUtils.writeStringToFile(featureFile,sequencePair.getFirst(),"utf-8");
            FileUtils.writeStringToFile(labelFile,sequencePair.getSecond().toString(),"utf-8");
        }
    }
}
