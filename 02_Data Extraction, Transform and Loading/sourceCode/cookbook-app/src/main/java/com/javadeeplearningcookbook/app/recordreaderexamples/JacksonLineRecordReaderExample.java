package com.javadeeplearningcookbook.app.recordreaderexamples;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.impl.jackson.FieldSelection;
import org.datavec.api.records.reader.impl.jackson.JacksonLineRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.shade.jackson.core.JsonFactory;
import org.nd4j.shade.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;

public class JacksonLineRecordReaderExample {
    public static void main(String[] args) throws IOException, InterruptedException {

        FieldSelection fieldSelection = new FieldSelection.Builder()
                                  .addField("sepalLength")
                                  .addField("sepalWidth")
                                  .addField("petalLength")
                                  .addField("petalWidth")
                                  .build();
        JacksonLineRecordReader jacksonLineRecordReader = new JacksonLineRecordReader(fieldSelection,new ObjectMapper(new JsonFactory()));
        Configuration configuration = new Configuration();
        configuration.set(JacksonLineRecordReader.LABELS,"species");
        jacksonLineRecordReader.setConf(configuration);
        jacksonLineRecordReader.initialize(new FileSplit(new File("D:/storage/irisdata.txt")));
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(jacksonLineRecordReader,5);
        System.out.println(dataSetIterator.totalOutcomes());



    }
}
