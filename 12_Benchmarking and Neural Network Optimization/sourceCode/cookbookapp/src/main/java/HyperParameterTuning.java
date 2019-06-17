import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.arbiter.MultiLayerSpace;
import org.deeplearning4j.arbiter.conf.updater.AdamSpace;
import org.deeplearning4j.arbiter.layers.DenseLayerSpace;
import org.deeplearning4j.arbiter.layers.OutputLayerSpace;
import org.deeplearning4j.arbiter.optimize.api.CandidateGenerator;
import org.deeplearning4j.arbiter.optimize.api.ParameterSpace;
import org.deeplearning4j.arbiter.optimize.api.data.DataSetIteratorFactoryProvider;
import org.deeplearning4j.arbiter.optimize.api.data.DataSource;
import org.deeplearning4j.arbiter.optimize.api.saving.ResultSaver;
import org.deeplearning4j.arbiter.optimize.api.score.ScoreFunction;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxCandidatesCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.MaxTimeCondition;
import org.deeplearning4j.arbiter.optimize.api.termination.TerminationCondition;
import org.deeplearning4j.arbiter.optimize.config.OptimizationConfiguration;
import org.deeplearning4j.arbiter.optimize.generator.RandomSearchGenerator;
import org.deeplearning4j.arbiter.optimize.parameter.continuous.ContinuousParameterSpace;
import org.deeplearning4j.arbiter.optimize.parameter.integer.IntegerParameterSpace;
import org.deeplearning4j.arbiter.optimize.runner.IOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.LocalOptimizationRunner;
import org.deeplearning4j.arbiter.optimize.runner.listener.impl.LoggingStatusListener;
import org.deeplearning4j.arbiter.saver.local.FileModelSaver;
import org.deeplearning4j.arbiter.scoring.impl.EvaluationScoreFunction;
import org.deeplearning4j.arbiter.task.MultiLayerNetworkTaskCreator;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;
import java.util.concurrent.TimeUnit;

public class HyperParameterTuning {
    public static final int labelIndex = 11;  // consider index 0 to 11  for input
    public static final int numClasses = 1;
    public static void main(String[] args) {

        ParameterSpace<Double> learningRateParam = new ContinuousParameterSpace(0.0001,0.01);
        ParameterSpace<Integer> layerSizeParam = new IntegerParameterSpace(5,11);
        MultiLayerSpace hyperParamaterSpace = new MultiLayerSpace.Builder()
                .updater(new AdamSpace(learningRateParam))
                //  .weightInit(WeightInit.DISTRIBUTION).dist(new LogNormalDistribution())
                .addLayer(new DenseLayerSpace.Builder()
                        .activation(Activation.RELU)
                        .nIn(11)
                        .nOut(layerSizeParam)
                        .build())
                .addLayer(new DenseLayerSpace.Builder()
                        .activation(Activation.RELU)
                        .nIn(layerSizeParam)
                        .nOut(layerSizeParam)
                        .build())
                .addLayer(new OutputLayerSpace.Builder()
                        .activation(Activation.SIGMOID)
                        .lossFunction(LossFunctions.LossFunction.XENT)
                        .nOut(1)
                        .build())
                .build();

        Map<String,Object> dataParams = new HashMap<>();
        dataParams.put("batchSize",new Integer(10));

        Map<String,Object> commands = new HashMap<>();
        commands.put(DataSetIteratorFactoryProvider.FACTORY_KEY,ExampleDataSource.class.getCanonicalName());

        CandidateGenerator candidateGenerator = new RandomSearchGenerator(hyperParamaterSpace,dataParams);

        Properties dataSourceProperties = new Properties();
        dataSourceProperties.setProperty("minibatchSize", "64");

        ResultSaver modelSaver = new FileModelSaver("resources/");
        ScoreFunction scoreFunction = new EvaluationScoreFunction(org.deeplearning4j.eval.Evaluation.Metric.ACCURACY);


        TerminationCondition[] conditions = {
                new MaxTimeCondition(120, TimeUnit.MINUTES),
                new MaxCandidatesCondition(30)

        };

        OptimizationConfiguration optimizationConfiguration = new OptimizationConfiguration.Builder()
                .candidateGenerator(candidateGenerator)
                .dataSource(ExampleDataSource.class,dataSourceProperties)
                .modelSaver(modelSaver)
                .scoreFunction(scoreFunction)
                .terminationConditions(conditions)
                .build();

        IOptimizationRunner runner = new LocalOptimizationRunner(optimizationConfiguration,new MultiLayerNetworkTaskCreator());
        //Uncomment this if you want to store the model.
        // StatsStorage ss = new FileStatsStorage(new File("HyperParamOptimizationStats.dl4j"));
        runner.addListeners(new LoggingStatusListener()); //new ArbiterStatusListener(ss)
        runner.execute();

        //Print the best hyper params

        double bestScore = runner.bestScore();
        int bestCandidateIndex = runner.bestScoreCandidateIndex();
        int numberOfConfigsEvaluated = runner.numCandidatesCompleted();

        String s = "Best score: " + bestScore + "\n" +
                "Index of model with best score: " + bestCandidateIndex + "\n" +
                "Number of configurations evaluated: " + numberOfConfigsEvaluated + "\n";

        System.out.println(s);

    }


    public static class ExampleDataSource implements DataSource{

        private int minibatchSize;

        public ExampleDataSource(){

        }

        @Override
        public void configure(Properties properties) {
            this.minibatchSize = Integer.parseInt(properties.getProperty("minibatchSize", "16"));
        }

        @Override
        public Object trainData() {
            try{
                DataSetIterator iterator = new RecordReaderDataSetIterator(dataPreprocess(),minibatchSize,labelIndex,numClasses);
                return dataSplit(iterator).getTestIterator();
            }
            catch(Exception e){
                throw new RuntimeException();
            }
        }

        @Override
        public Object testData() {
            try{
                DataSetIterator iterator = new RecordReaderDataSetIterator(dataPreprocess(),minibatchSize,labelIndex,numClasses);
                return dataSplit(iterator).getTestIterator();
            }
            catch(Exception e){
                throw new RuntimeException();
            }
        }

        @Override
        public Class<?> getDataType() {
            return DataSetIterator.class;
        }

        public DataSetIteratorSplitter dataSplit(DataSetIterator iterator) throws IOException, InterruptedException {
            DataNormalization dataNormalization = new NormalizerStandardize();
            dataNormalization.fit(iterator);
            iterator.setPreProcessor(dataNormalization);
            DataSetIteratorSplitter splitter = new DataSetIteratorSplitter(iterator,1000,0.8);
            return splitter;
        }

        public RecordReader dataPreprocess() throws IOException, InterruptedException {
            //Schema Definitions
            Schema schema = new Schema.Builder()
                    .addColumnsString("RowNumber")
                    .addColumnInteger("CustomerId")
                    .addColumnString("Surname")
                    .addColumnInteger("CreditScore")
                    .addColumnCategorical("Geography",Arrays.asList("France","Spain","Germany"))
                    .addColumnCategorical("Gender",Arrays.asList("Male","Female"))
                    .addColumnsInteger("Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary","Exited").build();

            //Schema Transformation
            TransformProcess transformProcess = new TransformProcess.Builder(schema)
                    .removeColumns("RowNumber","Surname","CustomerId")
                    .categoricalToInteger("Gender")
                    .categoricalToOneHot("Geography")
                    .removeColumns("Geography[France]")
                    .build();

            //CSVReader - Reading from file and applying transformation
            RecordReader reader = new CSVRecordReader(1,',');
            reader.initialize(new FileSplit(new ClassPathResource("Churn_Modelling.csv").getFile()));
            RecordReader transformProcessRecordReader = new TransformProcessRecordReader(reader,transformProcess);
            return transformProcessRecordReader;
        }
    }
}
