package com.javadeeplearningcookbook.examples;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.NGramTokenizerFactory;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;

public class BasicLineIteratorExample {
    public static void main(String[] args) throws IOException {
        SentenceIterator iterator = new BasicLineIterator(new ClassPathResource("raw_sentences.txt").getFile());
        int count=0;
        while(iterator.hasNext()){
           iterator.nextSentence();
           count++;
        }
        System.out.println("count = "+count);
        iterator.reset();
        SentenceDataPreProcessor.setPreprocessor(iterator);
        while(iterator.hasNext()){
            System.out.println(iterator.nextSentence());
        }
        

    }
}
