package com.springdl4j.springdl4j.service;

import com.javadeeplearningcookbook.api.CustomerRetentionPredictionApi;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

@Service
public class CookBookServiceImpl implements CookBookService {

    @Override
    public List<String> generateStringOutput(MultipartFile multipartFile) throws IOException, InterruptedException {
        final List<String> results = new ArrayList<>();
        File convFile = File.createTempFile(multipartFile.getOriginalFilename(),null, new File(System.getProperty("user.dir")+"/"));
        multipartFile.transferTo(convFile);
        INDArray indArray = AnimalClassifierAPI.generateOutput(convFile);
        for(int i=0; i<indArray.rows();i++){
            if(indArray.getDouble(i,0)>indArray.getDouble(i,1)){
                results.add("Customer "+(i+1)+"-> Happy Customer \n");
            }
            else{
                results.add("Customer "+(i+1)+"-> Unhappy Customer \n");
            }
        }
        convFile.deleteOnExit();

        return results;
    }
}
