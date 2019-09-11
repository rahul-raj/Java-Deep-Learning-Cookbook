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

    public static void main(String[] args){
        try {
            SequenceRecordReader trainFeatures = new CSVSequenceRecordReader(1);
            trainFeatures.initialize(new NumberedFileInputSplit(new File("{PATH-TO-FEATURES}/%d.csv").getPath(),1,4));
            SequenceRecordReader trainLabels = new CSVSequenceRecordReader(1);
            trainLabels.initialize(new NumberedFileInputSplit(new File("{PATH-TO-LABELS}/%d.csv").getPath(),1,4));

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
        } catch(RuntimeException e){
            System.out.println("Please provide proper file path for the dataset in place of: PATH-TO-FEATURES & PATH-TO-LABELS ");
        }catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }
}
