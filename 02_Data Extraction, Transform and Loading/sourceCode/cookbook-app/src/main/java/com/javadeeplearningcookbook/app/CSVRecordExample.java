package com.javadeeplearningcookbook.app;

import org.datavec.api.io.WritableConverter;
import org.datavec.api.io.converters.SelfWritableConverter;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.sql.SQLOutput;
import java.util.Arrays;

/**
 * @Author Rahul Raj
 * DataVec Example
 *
 */
public class CSVRecordExample
{
    public static void main( String[] args ) throws IOException, InterruptedException {

        int numClasses = 2;
        int batchSize = 8;

        File file = new File("titanic.csv");
        RecordReader recordReader = new CSVRecordReader(1,',');
        recordReader.initialize(new FileSplit(file));
        WritableConverter writableConverter = new SelfWritableConverter();

        Schema schema = new Schema.Builder()
                .addColumnInteger("Survived")
                .addColumnCategorical("Pclass", Arrays.asList("1","2","3"))
                .addColumnString("Name")
                .addColumnCategorical("Sex", Arrays.asList("male","female"))
                .addColumnsInteger("Age","Siblings/Spouses Aboard","Parents/Children Aboard")
                .addColumnDouble("Fare")
                .build();
        DataSetIterator dataSetIterator = new RecordReaderDataSetIterator(recordReader,writableConverter,8,1,7,2,-1,true);
        System.out.println("args = [" + dataSetIterator.totalOutcomes() + "]");
    }
}
