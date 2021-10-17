Please use the recommended data structure as follows:

```bash
data
    ├── ground_truth
    │   ├── groundtruth_2012-01-08.csv
    │   ├── groundtruth_2012-01-15.csv
    │   └── ...
    ├── ground_truth_cov
    │   ├── cov_2012-01-08.csv
    │   ├── cov_2012-01-15.csv
    │   └── ...
    ├── sensor_data
    │   ├── 2012-01-08_sen
    │   |    ├── gps.csv
    |   |    ├── gps_rtk.csv
    |   |    └── ...
    │   |── 2012-01-15_sen
    │   |    ├── gps.csv
    |   |    ├── gps_rtk.csv
    |   |    └── ...
    │   └── ...
    └── velodyne_data
        ├── 2012-01-08_vel
        |    ├── velodyne_sync
        |    |    └── ...
        |    └── velodyne_hits.bin
        |── 2012-01-15_vel
        |    ├── velodyne_sync
        |    |    └── ...
        |    └── velodyne_hits.bin
        └── ... 
```

