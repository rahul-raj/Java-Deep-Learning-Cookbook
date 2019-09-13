package com.javadeeplearningcookbook.app.recordreaderexamples;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.doubletransform.ConvertToDouble;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class TransformProcessRecordReaderExample {
    public static void main(String[] args) throws IOException, InterruptedException {
        try {
            Schema schema  =  new Schema.Builder()
                                        .addColumnsString("Name", "Subject")
                                        .addColumnInteger("Score")
                                        .addColumnCategorical("Grade", Arrays.asList("A","B","C","D"))
                                        .addColumnInteger("Passed").build();

            TransformProcess transformProcess = new TransformProcess.Builder(schema)
                                                      .removeColumns("Name")
                                                      .transform(new ConvertToDouble("Score"))
                                                      .categoricalToInteger("Grade").build();
            RecordReader recordReader = new CSVRecordReader(1, ',');
            recordReader.initialize(new FileSplit(new File("Path/to/transform-data.csv")));
            RecordReader transformRecordReader = new TransformProcessRecordReader(recordReader,transformProcess);
            System.out.println(transformRecordReader.next().get(0).toString());
        } catch(IllegalArgumentException e){
            System.out.println("Please provide proper directory path to transform-data.csv in place of: Path/to/transform-data.csv");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

    }
}
