package com.javadeeplearningcookbook.examples;

import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;

public class UimaSentenceIteratorExample {
    public static void main(String[] args) throws Exception {
        SentenceIterator iterator = UimaSentenceIterator.createWithPath("files/");
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
