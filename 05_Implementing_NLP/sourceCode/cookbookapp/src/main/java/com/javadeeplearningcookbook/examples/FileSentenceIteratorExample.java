package com.javadeeplearningcookbook.examples;

import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;

public class FileSentenceIteratorExample {
    public static void main(String[] args) throws IOException {
        SentenceIterator iterator = new FileSentenceIterator(new ClassPathResource("files/").getFile());
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
