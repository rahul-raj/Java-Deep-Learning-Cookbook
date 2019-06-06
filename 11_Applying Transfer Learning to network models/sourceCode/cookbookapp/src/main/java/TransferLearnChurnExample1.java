import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.TransferLearning;

import java.io.File;
import java.io.IOException;

public class TransferLearnChurnExample1 {
    private static final long seed = 12345;
    public static void main(String[] args) throws IOException {
        File savedLocation = new File("model.zip");      //Where to save the network. Note: the file is in .zip format - can be opened externally
        boolean saveUpdater = true;
        MultiLayerNetwork restored = MultiLayerNetwork.load(savedLocation, saveUpdater);



        MultiLayerNetwork newModel = new TransferLearning.Builder(restored)
                                          .setFeatureExtractor(1)
                                          .build();
    }
}
