# **File Information for NASA C-MAPSS 2 Aircraft Engine Run-to-Failure Dataset**

This comprehensive file guide will help you navigate the **NASA C-MAPSS 2 Aircraft Engine Run-to-Failure Dataset Under Real Flight Conditions**, which provides synthetic run-to-failure trajectories for a small fleet of **9 turbofan engines** under realistic flight conditions.\[2\]

## **Dataset Overview**

This enhanced dataset contains **5.3M samples for training and 1.2M samples for testing** with a sampling rate of **1Hz**, making it significantly larger than the original C-MAPSS dataset. The data represents realistic flight conditions recorded from commercial aircraft operations.\[2\]

## **Training Data Files**

Based on the visible file structure in your dataset, here are the main data files:\[1\]

### **N-CMAPSS\_DS01-005.h5 through N-CMAPSS\_DS0Xd-010.h5**

These are **HDF5 format files** containing the actual engine degradation data. The naming convention indicates:

* **DS01-DS0X**: Different dataset identifiers for various engines/scenarios  
* **h5 extension**: HDF5 format for efficient storage of large multidimensional arrays  
* **File sizes ranging from \~2.3GB to \~3.6GB**: Indicating substantial data volume per engine

## **Fleet Composition and Structure**

### **Training Units (6 engines)\[2\]**

* **Units 2, 5, 10**: Subject to **HPT (High Pressure Turbine) efficiency degradation**  
* **Units 16, 18, 20**: Subject to **complex failure mode affecting LPT (Low Pressure Turbine) efficiency and flow combined with HPT efficiency degradation**

### **Test Units (3 engines)\[2\]**

* **Units 11, 14, 15**: All subject to **HPT+LPT complex failure mode**  
* **Unit 14**: Operates shorter and lower altitude flights with highest number of flight cycles despite lowest flight time  
* **Unit 15**: Also operates shorter and lower altitude flights compared to training units

## **Data Structure Details**

Each data file contains **45 variables total**:\[2\]

### **Scenario Descriptors (4 variables)**

1. **alt**: Altitude (ft)  
2. **Mach**: Flight Mach number  
3. **TRA**: Throttle-resolver angle (%)  
4. **T2**: Total temperature at fan inlet (°R)

### **Sensor Measurements (13 variables)**

1. **Wf**: Fuel flow (pps)  
2. **Nf**: Physical fan speed (rpm)  
3. **Nc**: Physical core speed (rpm)  
4. **T24**: Total temperature at LPC outlet (°R)  
5. **T30**: Total temperature at HPC outlet (°R)  
6. **T48**: Total temperature at HPT outlet (°R)  
7. **T50**: Total temperature at LPT outlet (°R)  
8. **P15**: Total pressure in bypass-duct (psia)  
9. **P21**: Total pressure at fan outlet (psia)  
10. **P24**: Total pressure at LPC outlet (psia)  
11. **Ps30**: Static pressure at HPC outlet (psia)  
12. **P40**: Total pressure at burner outlet (psia)  
13. **P50**: Total pressure at LPT outlet (psia)

### **Virtual Sensors (18 variables)**

1. **T40**: Total temperature at burner outlet (°R)  
2. **P30**: Total pressure at HPC outlet (psia)  
    20-35. **Additional virtual sensor measurements** including flow rates, engine pressure ratios, stall margins, corrected speeds, and efficiency ratios

### **Model Health Parameters (10 variables)**

36-45. **Component efficiency and flow modifiers** for Fan, LPC, HPC, HPT, and LPT components

## 

## 

## **Dataset Size and Complexity**

| Unit | Data Size | Transition Time (ts) | End-of-Life Time (tEOL) | Failure Mode |
| :---- | :---- | :---- | :---- | :---- |
| 2 | 0.85M samples | 17 cycles | 75 cycles | HPT |
| 5 | 1.03M samples | 17 cycles | 89 cycles | HPT |
| 10 | 0.95M samples | 17 cycles | 82 cycles | HPT |
| 16 | 0.77M samples | 16 cycles | 63 cycles | HPT+LPT |
| 18 | 0.89M samples | 17 cycles | 71 cycles | HPT+LPT |
| 20 | 0.77M samples | 17 cycles | 66 cycles | HPT+LPT |
| 11 | 0.66M samples | 19 cycles | 59 cycles | HPT+LPT |
| 14 | 0.16M samples | 36 cycles | 76 cycles | HPT+LPT |
| 15 | 0.43M samples | 24 cycles | 67 cycles | HPT+LPT |

