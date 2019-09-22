package com.javadeeplearningcookbook.examples;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.io.File;
import java.util.Arrays;

public class GoogleNewsVectorExample {

    public static void main(String[] args) {
        try{
            File file = new File("{PATH-TO-GOOGLE-WORD-VECTOR}");
            Word2Vec model = WordVectorSerializer.readWord2VecModel(file);
            System.out.println(Arrays.asList(model.wordsNearest("season",10)));
        } catch(ND4JIllegalStateException e){
            System.out.println("Please provide proper directory path in place of: PATH-TO-GOOGLE-WORD-VECTOR");
        }
    }
}
