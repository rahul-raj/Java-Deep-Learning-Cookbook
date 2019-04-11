package com.javadeeplearningcookbook.api;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;

import java.io.File;
import java.io.IOException;

public class AnimalClassifierAPI {
    public static INDArray generateOutput(File file) throws IOException, InterruptedException {
        final File modelFile = new File("cnntrainedmodel.zip");
        final MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        final RecordReader imageRecordReader = generateReader(file);
        final NormalizerStandardize normalizerStandardize = ModelSerializer.restoreNormalizerFromFile(modelFile);
        final DataSetIterator dataSetIterator = new RecordReaderDataSetIterator.Builder(imageRecordReader,1).build();
        normalizerStandardize.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(normalizerStandardize);
        return model.output(dataSetIterator);
    }

    private static RecordReader generateReader(File file) throws IOException, InterruptedException {
        final RecordReader recordReader = new ImageRecordReader(30,30,3);
        final InputSplit inputSplit = new FileSplit(file);
        recordReader.initialize(inputSplit);
        return recordReader;
    }

    public static void main(String[] args) throws IOException, InterruptedException {
        final File file = new File("sample.png");
        final INDArray results = generateOutput(file);
    }
}



