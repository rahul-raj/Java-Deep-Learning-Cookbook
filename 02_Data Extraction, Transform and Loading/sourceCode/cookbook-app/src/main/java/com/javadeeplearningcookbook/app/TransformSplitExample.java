package com.javadeeplearningcookbook.app;

import org.datavec.api.split.CollectionInputSplit;
import org.datavec.api.split.TransformSplit;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Arrays;
import java.util.List;

public class TransformSplitExample {
    public static void main(String[] args) throws URISyntaxException {
       TransformSplit.URITransform uriTransform = URI::normalize;
        List<URI> uriList = Arrays.asList(new URI("file://storage/examples/./cats.txt"),
                                          new URI("file://storage/examples//dogs.txt"),
                                          new URI("file://storage/./examples/bear.txt"));
        TransformSplit transformSplit = new TransformSplit(new CollectionInputSplit(uriList),uriTransform);
        transformSplit.locationsIterator().forEachRemaining(System.out::println);

        //search and replace example
        List<URI> uriReplaceList = Arrays.asList(new URI("file://storage/examples/0/inputs.txt"),
                                                 new URI("file://storage/examples/1/inputs.txt"),
                                                 new URI("file://storage/examples/2/inputs.txt"));
        TransformSplit transformReplaceSplit = TransformSplit.ofSearchReplace(
                                                                  new CollectionInputSplit(uriReplaceList),
                                                                        "inputs","outputs");

        transformReplaceSplit.locationsIterator().forEachRemaining(System.out::println);




    }
}
