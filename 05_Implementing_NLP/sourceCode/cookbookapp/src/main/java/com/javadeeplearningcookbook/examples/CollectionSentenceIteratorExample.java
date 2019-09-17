package com.javadeeplearningcookbook.examples;

import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

public class CollectionSentenceIteratorExample {
    public static void main(String[] args) throws IOException {
        List<String> sentences = Arrays.asList(
                "No ,  he says now .",
                "And what did he do ?",
                "The money 's there .",
                "That was less than a year ago .",
                "But he made only the first .",
                "There 's still time for them to do it .",
                "But he should nt have .",
                " They have to come down to the people .",
                "I do nt know where that is .",
                "No , I would nt .",
                "Who Will It Be ?",
                "And no , I was not the one ."
        );
        SentenceIterator iterator = new CollectionSentenceIterator(sentences);
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
