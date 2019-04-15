package com.javadeeplearningcookbook.examples;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.EndingPreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.io.ClassPathResource;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Collection;

public class Word2VecModelExample {
    private static Logger log = LoggerFactory.getLogger(Word2VecModelExample.class);
    public static void main(String[] args) throws IOException {
        SentenceIterator iterator = new LineSentenceIterator(new ClassPathResource("raw_sentences_large.txt").getFile());
        SentenceDataPreProcessor.setPreprocessor(iterator);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new EndingPreProcessor());

        Word2Vec model = new Word2Vec.Builder()
                                        .iterate(iterator)
                                        .tokenizerFactory(tokenizerFactory)
                                        .minWordFrequency(5)
                                        .layerSize(100)
                                        .seed(42)
                                        .windowSize(5)
                                        .build();
        log.info("Fitting Word2Vec model....");
        model.fit();

        Collection<String> words = model.wordsNearest("season",10);
        for(String word: words){
            System.out.println(word+ " ");
        }
        double cosSimilarity = model.similarity("season","program");
        System.out.println(cosSimilarity);
    }
}
