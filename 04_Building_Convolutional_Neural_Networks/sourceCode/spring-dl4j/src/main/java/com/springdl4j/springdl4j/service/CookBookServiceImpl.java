package com.springdl4j.springdl4j.service;

import com.javadeeplearningcookbook.api.ImageClassifierAPI;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

@Service
public class CookBookServiceImpl implements CookBookService {

    @Override
    public List<String> generateStringOutput(MultipartFile multipartFile, String modelFilePath) throws IOException, InterruptedException {
        final List<String> results = new ArrayList<>();
        File convFile = File.createTempFile(multipartFile.getOriginalFilename(),null, new File(System.getProperty("user.dir")+"/"));
        multipartFile.transferTo(convFile);
        INDArray indArray = ImageClassifierAPI.generateOutput(convFile,modelFilePath);
        DecimalFormat df2 = new DecimalFormat("#.####");
       for(int i=0; i<indArray.rows();i++){
           String result="Image "+String.valueOf(i)+"->>>>>";
            for(int j=0;j<indArray.columns();j++){
                result+="\n Category "+j+": "+df2.format(indArray.getDouble(i,j)*100)+"%,   ";
            }
            result+="\n\n";
            results.add(result);

        }
        convFile.deleteOnExit();

        return results;
    }
}
