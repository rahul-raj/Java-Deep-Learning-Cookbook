package com.javadeeplearningcookbook.app.recordreaderexamples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

/**
 * @Author Rahul Raj
 * DataVec Example
 *
 */
public class CSVRecordReaderExample
{
    public static void main( String[] args ) throws IOException, InterruptedException {

        int numClasses = 2;
        int batchSize = 8;

        File file = new File("titanic.csv");
        RecordReader recordReader = new CSVRecordReader(1,',');
        recordReader.initialize(new FileSplit(file));
        // WritableConverter writableConverter = new SelfWritableConverter();

        Schema schema = new Schema.Builder()
                .addColumnInteger("Survived")
                .addColumnCategorical("Pclass", Arrays.asList("1","2","3"))
                .addColumnString("Name")
                .addColumnCategorical("Sex", Arrays.asList("male","female"))
                .addColumnsInteger("Age","Siblings/Spouses Aboard","Parents/Children Aboard")
                .addColumnDouble("Fare")
                .build();
        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                                                .removeColumns("Name","Fare")
                                                .categoricalToInteger("Sex")
                                                .categoricalToOneHot("Pclass")
                                                .removeColumns("Pclass[1]")
                                                .build();

        RecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,transformProcess);
        //DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(transformProcessRecordReader,writableConverter,8,1,7,2,-1,true);
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator.Builder(transformProcessRecordReader,batchSize)
                                              .classification(0,numClasses)
                                              .build();
        System.out.println("Total number of possible labels = [" + dataSetIterator.totalOutcomes()+ "]");

    }
}
