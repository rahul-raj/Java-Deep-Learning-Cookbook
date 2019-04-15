package com.javadeeplearningcookbook.examples;

import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

public class SentenceDataPreProcessor {
    public static void setPreprocessor(SentenceIterator iterator){
        iterator.setPreProcessor(((String sentence) -> sentence.toLowerCase()));
    }
}
