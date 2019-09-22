import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIteratorSplitter;
import org.nd4j.linalg.dataset.ExistingMiniBatchDataSetIterator;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class DataSetIteratorHelper {

    private final static int labelIndex=11;
    private final static int batchSize=8;
    private final static int numClasses=2;
    private static final int featurizeExtractionLayer = 1;
    public static DataSetIterator trainIterator() throws IOException, InterruptedException {
        return createDataSetSplitter().getTrainIterator();
    }
    public static DataSetIterator testIterator() throws IOException, InterruptedException {
        return createDataSetSplitter().getTestIterator();
    }

    private static DataSetIteratorSplitter createDataSetSplitter() throws IOException, InterruptedException {
        final RecordReader recordReader = DataSetIteratorHelper.generateReader(new ClassPathResource("Churn_Modelling.csv").getFile());
        final DataSetIterator dataSetIterator = new RecordReaderDataSetIterator.Builder(recordReader,batchSize)
                .classification(labelIndex,numClasses)
                .build();
        final DataNormalization dataNormalization = new NormalizerStandardize();
        dataNormalization.fit(dataSetIterator);
        dataSetIterator.setPreProcessor(dataNormalization);
        final DataSetIteratorSplitter dataSetIteratorSplitter = new DataSetIteratorSplitter(dataSetIterator,1250,0.8);
        return dataSetIteratorSplitter;
    }
    private static Schema generateSchema(){
        final Schema schema;
        schema = new Schema.Builder()
                .addColumnString("RowNumber")
                .addColumnInteger("CustomerId")
                .addColumnString("Surname")
                .addColumnInteger("CreditScore")
                .addColumnCategorical("Geography", Arrays.asList("France","Germany","Spain"))
                .addColumnCategorical("Gender", Arrays.asList("Male","Female"))
                .addColumnsInteger("Age", "Tenure")
                .addColumnDouble("Balance")
                .addColumnsInteger("NumOfProducts","HasCrCard","IsActiveMember")
                .addColumnDouble("EstimatedSalary")
                .addColumnInteger("Exited")
                .build();
        return schema;

    }
    private static RecordReader applyTransform(RecordReader recordReader, Schema schema){
        final TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .removeColumns("RowNumber","CustomerId","Surname")
                .categoricalToInteger("Gender")
                .categoricalToOneHot("Geography")
                .removeColumns("Geography[France]")
                .build();
        final TransformProcessRecordReader transformProcessRecordReader = new TransformProcessRecordReader(recordReader,transformProcess);
        return  transformProcessRecordReader;

    }
    public static RecordReader generateReader(File file) throws IOException, InterruptedException {
        final RecordReader recordReader = new CSVRecordReader(1,',');
        recordReader.initialize(new FileSplit(file));
        final RecordReader transformProcessRecordReader=applyTransform(recordReader,generateSchema());
        return transformProcessRecordReader;
    }

    public static DataSetIterator trainIteratorFeaturized(){
        DataSetIterator trainIter = new ExistingMiniBatchDataSetIterator(new File("{PATH-TO-SAVE-TRAIN-SAMPLES}"),"churn-"+featurizeExtractionLayer+"-train-%d.bin");
        return new AsyncDataSetIterator(trainIter);

    }
    public static DataSetIterator testIteratorFeaturized(){
        DataSetIterator testIter = new ExistingMiniBatchDataSetIterator(new File("{PATH-TO-SAVE-TEST-SAMPLES}"),"churn-"+featurizeExtractionLayer+"-test-%d.bin");
        return new AsyncDataSetIterator(testIter);
    }
}
