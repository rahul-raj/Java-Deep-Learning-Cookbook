package com.javadeeplearningcookbook.examples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class CustomerRetentionPredictionExample {

    private static final Logger log = LoggerFactory.getLogger("com.javadeeplearningcookbook.examples.CustomerLossPrediction.class");

    private static Schema generateSchema(){
        final Schema schema = new Schema.Builder()
                                    .addColumnString("RowNumber")
                                    .addColumnInteger("CustomerId")
                                    .addColumnString("Surname")
                                    .addColumnInteger("CreditScore")
                                    .addColumnCategorical("Geography", Arrays.asList("France","Germany","Spain"))
                                    .addColumnCategorical("Gender", Arrays.asList("Male","Female"))
                                    .addColumnsInteger("Age", "Tenure")
                                    .addColumnDouble("Balance")
                                    .addColumnsInteger("NumOfProducts","HasCrCard","IsActiveMember")
                                    .addColumnDouble("EstimatedSalary")
                                    .addColumnInteger("Exited")
                                    .build();
        return schema;

    }

    private static RecordReader applyTransform(RecordReader recordReader, Schema schema){
        final TransformProcess transformProcess = new TransformProcess.Builder(schema)
                                                                    .removeColumns("RowNumber","CustomerId","Surname")
                                                                    .categoricalToInteger("Gender")
                                                                    .categoricalToOneHot("Geography")
                                                                    .removeColumns("Geography[France]")
                                                                    .build();
        final TransformProcessRecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,transformProcess);
        return  transformProcessRecordReader;

    }

    private static RecordReader generateReader(File file) throws IOException, InterruptedException {
        final RecordReader recordReader = new CSVRecordReader(1,',');
        recordReader.initialize(new FileSplit(file));
        final RecordReader transformProcessRecordReader=applyTransform(recordReader,generateSchema());
        return transformProcessRecordReader;
    }

    private static INDArray generateOutput(File file) throws IOException, InterruptedException {
        final File modelFile = new File("model.zip");
        final MultiLayerNetwork network = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        final RecordReader recordReader = generateReader(file);
        final INDArray array = RecordConverter.toArray(recordReader.next());
        final NormalizerStandardize normalizerStandardize = ModelSerializer.restoreNormalizerFromFile(modelFile);
        normalizerStandardize.transform(array);
        return network.output(array,false);

    }

    public static void main(String[] args) throws IOException, InterruptedException {

       final int labelIndex=11;
       final int batchSize=8;
       final int numClasses=2;
       final INDArray weightsArray = Nd4j.create(new double[]{0.57, 0.75});

       final RecordReader recordReader = generateReader(new ClassPathResource("Churn_Modelling.csv").getFile());
       final DataSetIterator dataSetIterator = new RecordReaderDataSetIterator.Builder(recordReader,batchSize)
                                                                .classification(labelIndex,numClasses)
                                                                .build();
       final DataNormalization dataNormalization = new NormalizerStandardize();
       dataNormalization.fit(dataSetIterator);
       dataSetIterator.setPreProcessor(dataNormalization);
       final DataSetIteratorSplitter dataSetIteratorSplitter = new DataSetIteratorSplitter(dataSetIterator,1250,0.8);

       log.info("Building Model------------------->>>>>>>>>");

        final MultiLayerConfiguration configuration = new NeuralNetConfiguration.Builder()
                                                                    .weightInit(WeightInit.RELU_UNIFORM)
                                                                    .updater(new Adam(0.015D))
                                                                    .list()
                                                                    .layer(new DenseLayer.Builder().nIn(11).nOut(6).activation(Activation.RELU).dropOut(0.9).build())
                                                                    .layer(new DenseLayer.Builder().nIn(6).nOut(6).activation(Activation.RELU).dropOut(0.9).build())
                                                                    .layer(new DenseLayer.Builder().nIn(6).nOut(4).activation(Activation.RELU).dropOut(0.9).build())
                                                                    .layer(new OutputLayer.Builder(new LossMCXENT(weightsArray)).nIn(4).nOut(2).activation(Activation.SOFTMAX).build())
                                                                    .build();

        final UIServer uiServer = UIServer.getInstance();
        final StatsStorage statsStorage = new InMemoryStatsStorage();

        final MultiLayerNetwork multiLayerNetwork = new MultiLayerNetwork(configuration);
        multiLayerNetwork.init();
        multiLayerNetwork.setListeners(new ScoreIterationListener(100),
                                       new StatsListener(statsStorage));
        uiServer.attach(statsStorage);
        multiLayerNetwork.fit(dataSetIteratorSplitter.getTrainIterator(),100);

        final Evaluation evaluation =  multiLayerNetwork.evaluate(dataSetIteratorSplitter.getTestIterator(),Arrays.asList("0","1"));
        System.out.println(evaluation.stats());

        final File file = new File("model.zip");
        ModelSerializer.writeModel(multiLayerNetwork,file,true);
        ModelSerializer.addNormalizerToModel(file,dataNormalization);

        //INDArray output = generateOutput(new File("test.csv"));

    }
}
