# CarpoolSim: a simulation framework to find carpool trips

The structure of the project is shown as follows.
```
.
├── carpoolsim  # contains Python code (called by Jupyter Notebooks)
├── data_inputs  # all input data
│   ├── ABM2020 203K
│   │   ├── 2020 links  # traffic network link file
│   │   ├── 2020 nodes with latlon  # traffic network node file
│   │   └── taz  # traffic analysis zone
│   ├── Park_and_Ride_locations  # a shapefile/csv of parking lots
│   └── gt_survey  # survey of trip demands
├── data_outputs  # store the outputs (and intermediate results)
│   ├── ABM2020 203K
│   ├── mode1_path_results  # path results of carpoolsim mode 1
│   ├── mode2_path_results  # path results of carpoolsim mode 2
│   ├── mode3_path_results  # path results of carpoolsim mode 3
│   ├── step1_gt_survey  # processed gt survey dataset
│   ├── step2_parking_lots  # processed parking lots dataset
│   └── step2_results  # detail carpool matching results
└── notebooks # all Jupyter notebooks to run the program
```

Require inputs (all within the "data inputs folder")
1. Three network files:
   1. traffic networks shapefile: a shapefile of traffic links 
   2. traffic nodes shapefile: a shapefile of traffic nodes 
   3. taz: traffic analysis zone that splits a metropolitan region to many small parts
2. One file for PNR carpool:
   1. Park_and_Ride_locations: Parking lots that can be used for Park and Ride trip
3. One file for traffic demand:
   1. gt_survey: a survey of **trip demands** providing origin, destination, depart time, etc.







