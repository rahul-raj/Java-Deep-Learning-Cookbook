package com.javadeeplearningcookbook.app.recordreaderexamples;

import org.datavec.api.records.reader.impl.jackson.FieldSelection;
import org.datavec.api.records.reader.impl.jackson.JacksonLineRecordReader;
import org.datavec.api.records.reader.impl.jackson.JacksonRecordReader;

public class JacksonLineRecordReaderExample {
    public static void main(String[] args) {

        FieldSelection fieldSelection = new FieldSelection.Builder()
                                  .addField("sepalLength")
                                  .addField("sepalWidth")
                                  .addField("petalLength")
                                  .addField("petalWidth")
                                  .addField("species")
                                  .build();
        //JacksonLineRecordReader jacksonLineRecordReader = new JacksonLineRecordReader();
        //JacksonLineRecordReader
    }
}
