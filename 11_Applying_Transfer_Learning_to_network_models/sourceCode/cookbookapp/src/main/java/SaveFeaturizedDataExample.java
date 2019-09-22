import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

public class SaveFeaturizedDataExample {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(SaveFeaturizedDataExample.class);
    private static final int featurizeExtractionLayer = 1;
    private static final long seed = 12345;

    public static void main(String[] args) throws IOException, InterruptedException {
        //Use the model file saved from chapter 3 example.
        File savedLocation = new File("{PATH-TO-MODEL-FILE}");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;

        try{
            MultiLayerNetwork oldModel = MultiLayerNetwork.load(savedLocation, saveUpdater);

            FineTuneConfiguration fineTuneConf = new FineTuneConfiguration.Builder()
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(new Nesterovs(5e-5))
                    .biasInit(0.001)
                    .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                    .l2(0.0001)
                    .weightInit(WeightInit.DISTRIBUTION)
                    .seed(seed)
                    .build();

            MultiLayerNetwork newModel = new TransferLearning.Builder(oldModel)
                    .fineTuneConfiguration(fineTuneConf)
                    .setFeatureExtractor(featurizeExtractionLayer)
                    .build();

            TransferLearningHelper transferLearningHelper = new TransferLearningHelper(oldModel);
            DataSetIterator trainIter = DataSetIteratorHelper.trainIterator();
            DataSetIterator testIter = DataSetIteratorHelper.testIterator();

            int trainDataSaved = 0;
            while(trainIter.hasNext()) {
                DataSet currentFeaturized = transferLearningHelper.featurize(trainIter.next());
                saveToDisk(currentFeaturized,trainDataSaved,true);
                trainDataSaved++;
            }

            int testDataSaved = 0;
            while(testIter.hasNext()) {
                DataSet currentFeaturized = transferLearningHelper.featurize(testIter.next());
                saveToDisk(currentFeaturized,testDataSaved,false);
                testDataSaved++;
            }
        }
        catch(FileNotFoundException e){
            System.out.println("Please provide file path in place of: PATH-TO-MODEL-FILE, PATH-TO-SAVE-TRAIN-SAMPLES & PATH-TO-SAVE-TEST-SAMPLES");

        }
    }

    private static void saveToDisk(DataSet currentFeaturized, int iterNum, boolean isTrain){
        File fileFolder = isTrain ? new File("{PATH-TO-SAVE-TRAIN-SAMPLES}"): new File("{PATH-TO-SAVE-TEST-SAMPLES}");
        if (iterNum == 0) {
            fileFolder.mkdirs();
        }
        String fileName = "churn-" + featurizeExtractionLayer + "-";
        fileName += isTrain ? "train-" : "test-";
        fileName += iterNum + ".bin";
        currentFeaturized.save(new File(fileFolder,fileName));
        log.info("Saved " + (isTrain?"train ":"test ") + "dataset #"+ iterNum);
    }

}

