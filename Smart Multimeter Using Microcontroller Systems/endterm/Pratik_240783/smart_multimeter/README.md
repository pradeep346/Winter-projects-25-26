# Smart Digital Multimeter | Simulation & Design

## Part 1: Overview
This project is a software simulation of an industry-grade digital multimeter capable of measuring Resistance, Capacitance, and Inductance across a 10^5 range. It features a custom auto-ranging engine equipped with a 3-sample hysteresis rule to automatically select the optimal measurement scale without user input. By dynamically adjusting internal reference components, the simulation accurately models Gaussian hardware noise while achieving an average accuracy of ~98%.

## Part 2: How to set it up
Run the following commands in your terminal to clone the repository and install the required dependencies:

```bash
git clone https://github.com/pratikb24/Smart_Digital_Multimeter.git
cd Smart_Digital_Multimeter
pip install -r requirements.txt
```

## Part 3: How to run the simulation
To execute the primary measurement engine, run:

```bash
python simulate.py
```
This script tests 50 test values spread across all 5 ranges for Resistance, Capacitance, and Inductance. It passes each true value through simulated hardware physics (adding 0.5% Gaussian noise) and the auto-ranging engine. It then prints a results table showing the measured value, active range, and error percentage, and generates the required accuracy and auto-range state plots.

## Part 4: Your results
By utilizing the auto-ranging engine to dynamically shift internal reference values, the multimeter maintains extreme accuracy across a massive range.

| Method                  | R Error | C Error | L Error |
|-------------------------|---------|---------|---------|
| Fixed-range (no auto)   | >15.0% | >20.0% | >20.0% |
| **Auto-ranging sim** | **0.71%** | **0.41%** | **0.94%** |


## Part 5: Known limitations
While this simulation accurately models ideal physical equations and baseline Gaussian noise, a real hardware implementation would introduce additional complexities. Physical builds experience ADC noise, operational amplifier offset voltages, temperature drift altering the reference resistor tolerances over time, and parasitic probe resistance.
