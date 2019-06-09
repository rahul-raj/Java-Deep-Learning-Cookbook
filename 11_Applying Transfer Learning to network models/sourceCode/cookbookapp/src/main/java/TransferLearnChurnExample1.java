import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Nesterovs;

import java.io.File;
import java.io.IOException;

public class TransferLearnChurnExample1 {
    private static final long seed = 12345;
    public static void main(String[] args) throws IOException {
        final int labelIndex=11;
        final int batchSize=8;
        final int numClasses=2;

        final RecordReader recordReader = generateReader(new ClassPathResource("Churn_Modelling.csv").getFile());
        final DataSetIterator dataSetIterator = new RecordReaderDataSetIterator.Builder(recordReader,batchSize)
                .classification(labelIndex,numClasses)
                .build();
        final DataNormalization dataNormalization = new NormalizerStandardize();
        dataNormalization.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(dataNormalization);
        final DataSetIteratorSplitter dataSetIteratorSplitter = new DataSetIteratorSplitter(dataSetIterator,1250,0.8);

        File savedLocation = new File("model.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;
        MultiLayerNetwork restored = MultiLayerNetwork.load(savedLocation, saveUpdater);
        System.out.println(restored.getLayerWiseConfigurations().toJson());

        FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Nesterovs(5e-5))
                .activation(Activation.RELU6)
                .biasInit(0.001)
                .dropOut(0.85)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .l2(0.0001)
                .weightInit(WeightInit.IDENTITY)
                .seed(seed)
                .build();

        MultiLayerNetwork newModel = new TransferLearning.Builder(restored)
                                          .fineTuneConfiguration(fineTuneConf)
                                          .setFeatureExtractor(1)
                                          .build();

    }
    
     private static RecordReader generateReader(File file) throws IOException, InterruptedException {
        final RecordReader recordReader = new CSVRecordReader(1,',');
        recordReader.initialize(new FileSplit(file));
        final RecordReader transformProcessRecordReader=applyTransform(recordReader,generateSchema());
        return transformProcessRecordReader;
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
}
