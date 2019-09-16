package com.springdl4j.springdl4j.service;

import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;


public interface CookBookService {
    List<String> generateStringOutput(MultipartFile multipartFile, String modelFilePath) throws IOException, InterruptedException;
}
