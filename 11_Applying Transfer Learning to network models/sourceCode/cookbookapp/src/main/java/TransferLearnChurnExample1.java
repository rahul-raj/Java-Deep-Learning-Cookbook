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
}
