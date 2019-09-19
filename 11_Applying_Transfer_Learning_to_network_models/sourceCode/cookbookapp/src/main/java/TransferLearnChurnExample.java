import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.transferlearning.TransferLearningHelper;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.slf4j.Logger;

import java.io.File;
import java.io.IOException;

public class TransferLearnChurnExample {
    private static final Logger log = org.slf4j.LoggerFactory.getLogger(TransferLearnChurnExample.class);
    private static final long seed = 12345;
    private static final int featurizeExtractionLayer = 1;
    private static final int nEpochs = 100;
    public static void main(String[] args) throws IOException, InterruptedException {

        File savedLocation = new File("model.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;
        MultiLayerNetwork oldModel = MultiLayerNetwork.load(savedLocation, saveUpdater);
        //System.out.println(restored.getLayerWiseConfigurations().toJson());

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

        TransferLearningHelper transferLearningHelper = new TransferLearningHelper(newModel);

        DataSetIterator trainIterFeaturized = DataSetIteratorHelper.trainIteratorFeaturized();
        DataSetIterator testIterFeaturized = DataSetIteratorHelper.testIteratorFeaturized();

        for(int i=0;i<nEpochs;i++) {
            if (i == 0) {
                Evaluation eval = transferLearningHelper.unfrozenMLN().evaluate(testIterFeaturized);
                log.info("Eval stats BEFORE fit.....");
                log.info(eval.stats()+"\n");
                testIterFeaturized.reset();
            }
            int iter = 0;
            while (trainIterFeaturized.hasNext()) {
                transferLearningHelper.fitFeaturized(trainIterFeaturized.next());
                if (iter % 10 == 0) {
                    log.info("Evaluate model at iter " + iter + " ....");
                    Evaluation eval = transferLearningHelper.unfrozenMLN().evaluate(testIterFeaturized);
                    log.info(eval.stats());
                    testIterFeaturized.reset();
                }
                iter++;
            }
            trainIterFeaturized.reset();
            log.info("Epoch #"+i+" complete");
        }



    }



}
