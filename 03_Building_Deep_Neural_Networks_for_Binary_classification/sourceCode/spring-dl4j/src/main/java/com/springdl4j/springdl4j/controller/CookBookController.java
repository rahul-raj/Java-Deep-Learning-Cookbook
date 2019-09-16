package com.springdl4j.springdl4j.controller;

import com.springdl4j.springdl4j.service.CookBookService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

@Controller
public class CookBookController {

    @Autowired
    CookBookService cookBookService;

    @Value("${modelFilePath}")
    private String modelFilePath;

    @GetMapping("/")
    public String main(final Model model){
        model.addAttribute("message", "Welcome to Java deep learning!");
        return "welcome";
    }

    @PostMapping("/")
    public String fileUpload(final Model model, final @RequestParam("uploadFile")MultipartFile multipartFile) throws IOException, InterruptedException {
        final List<String> results = cookBookService.generateStringOutput(multipartFile,modelFilePath);
        model.addAttribute("message", "Welcome to Java deep learning!");
        model.addAttribute("results",results);
        return "welcome";
    }
}
