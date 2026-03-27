package com.fc.wartungski;

import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;

public interface MachineRepository extends JpaRepository<Machine, Long> {
    List<Machine> findAllByOrderByTimestampDesc();
}