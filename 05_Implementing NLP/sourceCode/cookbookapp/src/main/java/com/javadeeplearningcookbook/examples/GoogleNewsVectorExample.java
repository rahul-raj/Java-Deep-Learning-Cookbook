package com.javadeeplearningcookbook.examples;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;

import java.io.File;
import java.util.Arrays;

public class GoogleNewsVectorExample {

    public static void main(String[] args) {
        File file = new File("C:/Users/Admin/Downloads/GoogleNews-vectors-negative300.bin.gz");
        Word2Vec model = WordVectorSerializer.readWord2VecModel(file);
        System.out.println(Arrays.asList(model.wordsNearest("season",10)));
    }
}
