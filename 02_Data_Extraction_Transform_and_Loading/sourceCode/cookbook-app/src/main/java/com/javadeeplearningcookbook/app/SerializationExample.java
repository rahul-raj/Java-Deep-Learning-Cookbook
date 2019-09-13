package com.javadeeplearningcookbook.app;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.transform.doubletransform.ConvertToDouble;

import java.util.Arrays;

public class SerializationExample {
    public static void main(String[] args) {
        Schema schema  =  new Schema.Builder()
                .addColumnsString("Name", "Subject")
                .addColumnInteger("Score")
                .addColumnCategorical("Grade", Arrays.asList("A","B","C","D"))
                .addColumnInteger("Passed").build();

        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .removeColumns("Name")
                .transform(new ConvertToDouble("Score"))
                .categoricalToInteger("Grade").build();

        String json = transformProcess.toJson();
        System.out.println(json);

        String yaml = transformProcess.toYaml();
        System.out.println(yaml);
    }
}
