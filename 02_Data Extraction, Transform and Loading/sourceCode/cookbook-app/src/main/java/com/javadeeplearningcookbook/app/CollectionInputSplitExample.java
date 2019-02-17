package com.javadeeplearningcookbook.app;

import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.FileSplit;

import java.io.File;

public class CollectionInputSplitExample {
    public static void main(String[] args) {
        FileSplit fileSplit = new FileSplit(new File("temp"));
        CollectionInputSplit collectionInputSplit = new CollectionInputSplit(fileSplit.locations());
        collectionInputSplit.locationsIterator().forEachRemaining(System.out::println);
    }
}
