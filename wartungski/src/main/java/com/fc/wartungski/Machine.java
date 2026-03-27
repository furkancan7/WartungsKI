package com.fc.wartungski;

import jakarta.persistence.*;
import lombok.Data;
import java.time.LocalDateTime;

@Entity
@Data
public class Machine {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String machineName;
    private String predictionResult;
    private double probability;
    private String confidence;
    private LocalDateTime timestamp;
}