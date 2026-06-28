package com.fc.wartungski;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Controller;
import org.springframework.ui.Model;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.client.RestTemplate;

import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.Map;

@Controller
public class controller {

    @Autowired
    private MachineRepository repository;

    private final String FastAPI_URL = "http://127.0.0.1:8000/predict/machine";
    @GetMapping("/")
    public String home(Model model) {
        model.addAttribute("history", repository.findAllByOrderByTimestampDesc());
        return "index";
    }

    @PostMapping("/predict")
    public String predict(
            @RequestParam String name,
            @RequestParam double air_temperature,
            @RequestParam double process_temperature,
            @RequestParam double rotational_speed,
            @RequestParam double torque,
            @RequestParam double tool_wear,
            Model model) {

        RestTemplate restTemplate = new RestTemplate();
        Map<String, Object> requestBody = new HashMap<>();

        requestBody.put("air_temperature", air_temperature);
        requestBody.put("process_temperature", process_temperature);
        requestBody.put("rotational_speed", rotational_speed);
        requestBody.put("torque", torque);
        requestBody.put("tool_wear", tool_wear);
        requestBody.put("type_L", 1);
        requestBody.put("type_M", 0);

        try {
            Map<String, Object> response = restTemplate.postForObject(FastAPI_URL, requestBody, Map.class);

            if (response != null) {
                model.addAttribute("predictionResult", response.get("recommendation"));
                model.addAttribute("probability", response.get("failure_probability"));
                model.addAttribute("confidence", response.get("confidence"));
                model.addAttribute("modelInfo", response.get("model"));
                model.addAttribute("machineName", name);

                Machine log = new Machine();
                log.setMachineName(name);
                log.setPredictionResult((String) response.get("recommendation"));

                Object prob = response.get("failure_probability");
                log.setProbability(prob instanceof Double ? (Double) prob : Double.parseDouble(prob.toString()));

                log.setConfidence((String) response.get("confidence"));
                log.setTimestamp(LocalDateTime.now());

                repository.save(log); // Veritabanına kaydeder
            }
        } catch (Exception e) {
            model.addAttribute("predictionResult", "Hata: Python servisine ulaşılamadı. " + e.getMessage());
            model.addAttribute("probability", 0.0);
        }

        model.addAttribute("history", repository.findAllByOrderByTimestampDesc());
        return "index";
    }
}