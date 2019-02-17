package com.javadeeplearningcookbook.app;

import org.datavec.api.split.NumberedFileInputSplit;

public class NumberedFileInputSplitExample {
    public static void main(String[] args) {
        NumberedFileInputSplit numberedFileInputSplit = new NumberedFileInputSplit("numberedfiles/file%d.txt",1,4);
        numberedFileInputSplit.locationsIterator().forEachRemaining(System.out::println);
    }
}
