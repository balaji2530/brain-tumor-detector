package com.example.image.classifier.service;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;

import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

@Service
public class ImageService {

    public String classifyImage(MultipartFile file) throws Exception {

        String uploadDir = System.getProperty("user.dir") + "/uploads/";

        File dir = new File(uploadDir);
        if (!dir.exists()) {
            dir.mkdirs();
        }

        String filePath = uploadDir + file.getOriginalFilename();
        File dest = new File(filePath);
        file.transferTo(dest);

        String pythonScript = System.getProperty("user.dir") + "/python-model/predict.py";

        ProcessBuilder pb = new ProcessBuilder(
                "python", pythonScript, filePath);

        pb.redirectErrorStream(true);
        Process process = pb.start();

        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));

        String line;
        String prediction = "";

        while ((line = reader.readLine()) != null) {

            System.out.println(line);

            if (line.contains("Prediction:")) {
                prediction = line;
            }
        }

        process.waitFor();

        if (prediction.isEmpty()) {
            return "Prediction not found";
        }

        return prediction;
    }
}