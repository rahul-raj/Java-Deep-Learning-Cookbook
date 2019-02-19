package com.javadeeplearningcookbook.app.recordreaderexamples;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class SequenceRecordReaderExample {

    /*
    * Download dataset -> https://drive.google.com/file/d/1T5Y5GOx9mui7h4U_nBO7k1rpI1K4Mclg/view?usp=sharing
    *
    * */
    public static void main(String[] args) throws IOException, InterruptedException {
        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1);
        trainFeatures.initialize(new NumberedFileInputSplit(new File("D:/storage/features/%d.csv").getPath(),1,4));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader(1);
        trainLabels.initialize(new NumberedFileInputSplit(new File("D:/storage/labels/%d.csv").getPath(),1,4));

        Schema featureSchema = new Schema.Builder()
                .addColumnCategorical("Pclass", Arrays.asList("1","2","3"))
                .addColumnString("Name")
                .addColumnCategorical("Sex", Arrays.asList("male","female"))
                .addColumnsInteger("Age","Siblings/Spouses Aboard","Parents/Children Aboard")
                .addColumnDouble("Fare")
                .build();
        TransformProcess featureTransformProcess = new TransformProcess.Builder(featureSchema)
                .removeColumns("Name","Fare")
                .categoricalToInteger("Sex")
                .categoricalToOneHot("Pclass")
                .removeColumns("Pclass[1]")
                .build();
        TransformProcessSequenceRecordReader featureRecordReader = new TransformProcessSequenceRecordReader(trainFeatures,featureTransformProcess);


        Schema labelSchema = new Schema.Builder()
                .addColumnInteger("Survived")
                .build();
        TransformProcess labelTransformProcess = new TransformProcess.Builder(labelSchema)
                           .build();
        TransformProcessSequenceRecordReader labelRecordReader = new TransformProcessSequenceRecordReader(trainLabels,labelTransformProcess);

        DataSetIterator trainIterator = new SequenceRecordReaderDataSetIterator(featureRecordReader,labelRecordReader,10,2,false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
        System.out.println(trainIterator.inputColumns());

    }
}
