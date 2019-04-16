package com.javadeeplearningcookbook.examples;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.BarnesHutTsne;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TSNEVisualizationExample {
    public static void main(String[] args) throws IOException {
        Nd4j.setDataType(DataBuffer.Type.DOUBLE);
        List<String> cacheList = new ArrayList<>();
        File file = new File("words.txt");
        String outputFile = "tsne-standard-coords.csv";
        Pair<InMemoryLookupTable,VocabCache> vectors = WordVectorSerializer.loadTxt(file);
        VocabCache cache = vectors.getSecond();
        INDArray weights = vectors.getFirst().getSyn0();

        for(int i=0;i<cache.numWords();i++){
            cacheList.add(cache.wordAtIndex(i));
        }

        BarnesHutTsne tsne = new BarnesHutTsne.Builder()
                                                .setMaxIter(100)
                                                .theta(0.5)
                                                .normalize(false)
                                                .learningRate(500)
                                                .useAdaGrad(false)
                                                .build();

        tsne.fit(weights);
        tsne.saveAsFile(cacheList,outputFile);

    }
}
