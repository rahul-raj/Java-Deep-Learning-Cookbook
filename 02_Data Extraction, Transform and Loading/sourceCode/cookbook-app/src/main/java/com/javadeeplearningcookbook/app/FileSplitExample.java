package com.javadeeplearningcookbook.app;

import org.datavec.api.split.FileSplit;

import java.io.File;

public class FileSplitExample {
    public static void main(String[] args) {
        String[] allowedFormats=new String[]{".JPEG"};
        //recursive -> true, so that it will check for all subdirectories
        FileSplit fileSplit = new FileSplit(new File("temp"),allowedFormats,true);
        fileSplit.locationsIterator().forEachRemaining(System.out::println);

    }
}
