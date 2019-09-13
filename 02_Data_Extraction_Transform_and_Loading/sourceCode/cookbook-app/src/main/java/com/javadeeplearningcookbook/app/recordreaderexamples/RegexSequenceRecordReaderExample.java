package com.javadeeplearningcookbook.app.recordreaderexamples;

import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.regex.RegexSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;

import java.io.IOException;

public class RegexSequenceRecordReaderExample {
    public static void main(String[] args){
        try {
            NumberedFileInputSplit fileSplit = new NumberedFileInputSplit("Path/to/logdata",
                    1,
                    20);
            String regex = "(\\d{2}/\\d{2}/\\d{2}) (\\d{2}:\\d{2}:\\d{2}) ([A-Z]) (.*)";

            SequenceRecordReader recordReader = new RegexSequenceRecordReader(regex,0);
            recordReader.initialize(fileSplit);
            //There are 10 sequences of files.  We are printing one of the sample sequence here
            System.out.println(recordReader.next().get(0).toString());
        } catch(RuntimeException e){
            System.out.println("Please provide proper directory path to logdata in place of: Path/to/logdata");
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }


}

