package com.javacookbook.app;

import com.beust.jcommander.JCommander;
import com.beust.jcommander.Parameter;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.launcher.SparkLauncher;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.loader.impl.RecordReaderFileBatchLoader;
import org.deeplearning4j.datasets.fetchers.TinyImageNetFetcher;
import org.deeplearning4j.datasets.iterator.impl.TinyImageNetDataSetIterator;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.ConvolutionMode;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.GlobalPoolingLayer;
import org.deeplearning4j.nn.conf.layers.LossLayer;
import org.deeplearning4j.nn.conf.layers.PoolingType;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.solvers.accumulation.encoding.threshold.AdaptiveThresholdAlgorithm;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.impl.graph.SparkComputationGraph;
import org.deeplearning4j.spark.parameterserver.training.SharedTrainingMaster;
import org.deeplearning4j.spark.util.SparkUtils;
import org.deeplearning4j.zoo.model.helper.DarknetHelper;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.ISchedule;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


/**
 * This Example demonstrate DL4J spark standalone application
 * Make sure to run PreprocessLocal or PreprocessSpark first.
 */
public class SparkExample {


    public static final Logger log = LoggerFactory.getLogger(SparkExample.class);

    @Parameter(names = {"--dataPath"}, description = "Path (on HDFS or similar) of data preprocessed by preprocessing script." +
            " See PreprocessLocal or PreprocessSpark", required = true)
    private String dataPath;


    @Parameter(names = {"--masterIP"}, description = "Master node IP Address", required = true)
    private String masterIP;

    public static void main(String[] args) throws Exception {

       new SparkExample().entryPoint(args);

    }

    protected void entryPoint(String[] args) throws Exception{
        int batchSize = 32;

        JCommander jcmdr = new JCommander(this);
        jcmdr.parse(args);

        SparkConf conf = new SparkConf();
        conf.setMaster("local[*]");
        conf.setAppName("DL4J Spark Imagenet Classifier");
        conf.set("spark.locality.wait","0");
        conf.set("spark.executor.extraJavaOptions","-Dorg.bytedeco.javacpp.maxbytes=6G -Dorg.bytedeco.javacpp.maxphysicalbytes=6G");
        conf.set(SparkLauncher.DRIVER_EXTRA_JAVA_OPTIONS,"-Dorg.bytedeco.javacpp.maxbytes=6G -Dorg.bytedeco.javacpp.maxphysicalbytes=6G");
        JavaSparkContext context = new JavaSparkContext(conf);


        VoidConfiguration voidConfiguration = VoidConfiguration.builder()
                .controllerAddress(masterIP)
                .unicastPort(40123)                          // Port number that should be open for IN/OUT communications on all Spark nodes
                /*  .networkMask("192.168.0.0/16")                   // Local network mask
                .controllerAddress("192.168.0.139")                // IP address of the master/driver node
                .meshBuildMode(MeshBuildMode.PLAIN)*/
                .build();

        TrainingMaster tm = new SharedTrainingMaster.Builder(voidConfiguration, batchSize)
                .rngSeed(12345)
                .collectTrainingStats(false)
                .batchSizePerWorker(batchSize)              // Minibatch size for each worker
                .thresholdAlgorithm(new AdaptiveThresholdAlgorithm(1E-3))     //Threshold algorithm determines the encoding threshold to be use.
                .workersPerNode(1)          // Workers per node
                .build();


        ISchedule lrSchedule = new MapSchedule.Builder(ScheduleType.EPOCH)
                .add(0, 8e-3)
                .add(1, 6e-3)
                .add(3, 3e-3)
                .add(5, 1e-3)
                .add(7, 5e-4).build();

        ComputationGraphConfiguration.GraphBuilder builder = new NeuralNetConfiguration.Builder()
                .convolutionMode(ConvolutionMode.Same)
                .l2(1e-4)
                .updater(new AMSGrad(lrSchedule))
                .weightInit(WeightInit.RELU)
                .graphBuilder()
                .addInputs("input")
                .setOutputs("output");

        /*

          We use DarknetHelper from DL4J model zoo. We use DarkNet CNN architecture for our tiny imagenet CNN

         */
        DarknetHelper.addLayers(builder, 0, 3, 3, 32, 0);     //64x64 out
        DarknetHelper.addLayers(builder, 1, 3, 32, 64, 2);    //32x32 out
        DarknetHelper.addLayers(builder, 2, 2, 64, 128, 0);   //32x32 out
        DarknetHelper.addLayers(builder, 3, 2, 128, 256, 2);   //16x16 out
        DarknetHelper.addLayers(builder, 4, 2, 256, 256, 0);   //16x16 out
        DarknetHelper.addLayers(builder, 5, 2, 256, 512, 2);   //8x8 out

        builder.addLayer("convolution2d_6", new ConvolutionLayer.Builder(1, 1)
                .nIn(512)
                .nOut(TinyImageNetFetcher.NUM_LABELS) // number of labels (classified outputs) = 200
                .weightInit(WeightInit.XAVIER)
                .stride(1, 1)
                .activation(Activation.IDENTITY)
                .build(), "maxpooling2d_5")
                .addLayer("globalpooling", new GlobalPoolingLayer.Builder(PoolingType.AVG).build(), "convolution2d_6")
                .addLayer("loss", new LossLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).activation(Activation.SOFTMAX).build(), "globalpooling")
                .setOutputs("loss");

        ComputationGraphConfiguration configuration = builder.build();
        SparkComputationGraph sparkNet = new SparkComputationGraph(context,configuration,tm);
        sparkNet.setListeners(new PerformanceListener(10, true));

        //Create data loader
        int imageHeightWidth = 64;      //64x64 pixel input
        int imageChannels = 3;          //RGB
        int numEpochs=100;
        PathLabelGenerator labelMaker = new ParentPathLabelGenerator();
        ImageRecordReader rr = new ImageRecordReader(imageHeightWidth, imageHeightWidth, imageChannels, labelMaker);
        rr.setLabels(new TinyImageNetDataSetIterator(1).getLabels());
        RecordReaderFileBatchLoader loader = new RecordReaderFileBatchLoader(rr, batchSize, 1, TinyImageNetFetcher.NUM_LABELS);
        loader.setPreProcessor(new ImagePreProcessingScaler());   //Scale 0-255 valued pixels to 0-1 range


        //Fit the network
        String trainPath = dataPath + (dataPath.endsWith("/") ? "" : "/") + "train";
        JavaRDD<String> pathsTrain = SparkUtils.listPaths(context, trainPath);
        for (int i = 0; i < numEpochs; i++) {
            log.info("--- Starting Training: Epoch {} of {} ---", (i + 1), numEpochs);
            sparkNet.fitPaths(pathsTrain, loader);
        }

        //Perform evaluation
        String testPath = dataPath + (dataPath.endsWith("/") ? "" : "/") + "test";
        JavaRDD<String> pathsTest = SparkUtils.listPaths(context, testPath);

        //Set up for top 5 accuracy
        Evaluation evaluation = new Evaluation(TinyImageNetDataSetIterator.getLabels(false), 5);
        evaluation = (Evaluation) sparkNet.doEvaluation(pathsTest, loader, evaluation)[0];
        log.info("Evaluation statistics: {}", evaluation.stats());

        log.info("----- Example Complete -----");


    }
}
