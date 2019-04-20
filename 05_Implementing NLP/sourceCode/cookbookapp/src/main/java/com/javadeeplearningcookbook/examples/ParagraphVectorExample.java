package com.javadeeplearningcookbook.examples;

import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.primitives.Pair;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class ParagraphVectorExample {
    private static Logger log = LoggerFactory.getLogger(ParagraphVectorExample.class);
    public static void main(String[] args) throws IOException {

        LabelAwareIterator labelAwareIterator = new FileLabelAwareIterator.Builder()
                                                        .addSourceFolder(new ClassPathResource("label").getFile())
                                                .build();
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());

        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                                                    .learningRate(0.025)
                                                    .minLearningRate(0.005)
                                                    .batchSize(1000)
                                                    .epochs(5)
                                                    .iterate(labelAwareIterator)
                                                    .trainWordVectors(true)
                                                    .tokenizerFactory(tokenizerFactory)
                                            .build();
        paragraphVectors.fit();

        ClassPathResource unClassifiedResource = new ClassPathResource("unlabeled");
        FileLabelAwareIterator unClassifiedIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(unClassifiedResource.getFile())
                .build();
        InMemoryLookupTable<VocabWord> lookupTable = (InMemoryLookupTable<VocabWord>)paragraphVectors.getLookupTable();


        while (unClassifiedIterator.hasNextDocument()) {
            LabelledDocument labelledDocument = unClassifiedIterator.nextDocument();
            List<String> documentAsTokens = tokenizerFactory.create(labelledDocument.getContent()).getTokens();
            VocabCache vocabCache = lookupTable.getVocab();
            AtomicInteger cnt = new AtomicInteger(0);
            for (String word: documentAsTokens) {
                if (vocabCache.containsWord(word)){
                    cnt.incrementAndGet();
                }
            }
            INDArray allWords = Nd4j.create(cnt.get(), lookupTable.layerSize());
            cnt.set(0);
            for (String word: documentAsTokens) {
                if (vocabCache.containsWord(word))
                    allWords.putRow(cnt.getAndIncrement(), lookupTable.vector(word));
            }
            INDArray documentVector = allWords.mean(0);

            List<String> labels = labelAwareIterator.getLabelsSource().getLabels();

            List<Pair<String, Double>> result = new ArrayList<>();
            for (String label: labels) {
                INDArray vecLabel = lookupTable.vector(label);
                if (vecLabel == null){
                    throw new IllegalStateException("Label '"+ label+"' has no known vector!");
                }
                double sim = Transforms.cosineSim(documentVector, vecLabel);
                result.add(new Pair<String, Double>(label, sim));
            }

            for (Pair<String, Double> score: result) {
                log.info("        " + score.getFirst() + ": " + score.getSecond());
            }

        }
    }
}
