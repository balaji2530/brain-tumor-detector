package com.example.image.classifier.controller;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import com.example.image.classifier.service.ImageService;

@RestController
@RequestMapping("/api")
public class ImageController {

    @Autowired
    private ImageService imageService;

    @PostMapping("/upload")
    public String uploadImage(@RequestParam("file") MultipartFile file) {

        try {
            return imageService.classifyImage(file);
        } catch (Exception e) {
            return "Error: " + e.getMessage();
        }

    }
}