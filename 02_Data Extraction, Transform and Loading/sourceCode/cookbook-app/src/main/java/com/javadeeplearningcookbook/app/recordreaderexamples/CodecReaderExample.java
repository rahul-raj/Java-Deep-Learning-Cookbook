package com.javadeeplearningcookbook.app.recordreaderexamples;

import org.datavec.api.conf.Configuration;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.datavec.codec.reader.CodecRecordReader;
import org.datavec.codec.reader.NativeCodecRecordReader;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class CodecReaderExample {
    public static void main(String[] args) throws IOException, InterruptedException {
        SequenceRecordReader codecRecordReader = new NativeCodecRecordReader();
        Configuration conf = new Configuration();
        conf.set(CodecRecordReader.RAVEL, "true");
        conf.set(CodecRecordReader.START_FRAME, "2");
        conf.set(CodecRecordReader.TOTAL_FRAMES, "10");
        conf.set(CodecRecordReader.ROWS, "80");
        conf.set(CodecRecordReader.COLUMNS, "46");
        conf.set(CodecRecordReader.VIDEO_DURATION,"30");
        codecRecordReader.initialize(new FileSplit(new File("D:/storage/Wildlife.mp4")));
        codecRecordReader.setConf(conf);
        List<List<Writable>> list =  codecRecordReader.sequenceRecord();
        list.listIterator().forEachRemaining(el->System.out.println(el.size()));

    }
}
