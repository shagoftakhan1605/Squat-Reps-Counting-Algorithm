# Real-time-squat-rep-analysis-using-Computer-Vision-and-Time-Series

This project aims to analyze squats using real-time body landmark detection and distance measurements, leveraging MediaPipe and OpenCV for computer vision tasks. It records the necessary data, performs time series analysis, and visualizes the results to differentiate between correct and faulty squats.

## Table of Contents
- [Features](#features)
- [Results](#results)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Real-Time Squat Analysis](#real-time-squat-analysis)
  - [Time Series Analysis](#time-series-analysis)
- [Dataset](#dataset)
- [Model Building](#model-building)
- [Time Series Analysis and Visualization](#time-series-analysis-and-visualization)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Features
- Real-time detection of body landmarks using MediaPipe.
- Calculation of distances between body parts.
- Recording of distance data for time series analysis.
- Visualization of distance data over time.
- Comprehensive analysis to identify correct and faulty squats.

## Results
![3](https://github.com/user-attachments/assets/32ec32d0-ddb7-40e0-b103-34ca61c40895)
![2](https://github.com/user-attachments/assets/5114b10e-2cb0-43c9-83d2-30c4c3c14f37)
![1](https://github.com/user-attachments/assets/3ecc4c9d-4a66-4653-bc7d-16ecc10d977c)
![4](https://github.com/user-attachments/assets/d576ccd9-742c-423e-9a50-5d7548943aad)


## Requirements
- Python 3.x
- TensorFlow
- OpenCV
- Matplotlib
- MediaPipe
- pandas

## Installation
Install the required libraries:
  ```bash
  pip install tensorflow opencv-python matplotlib mediapipe pandas
  ```

## Usage

### Real-Time Squat Analysis
1. Run the real-time squat analysis script:
    ```bash
    python real_time_squat_analysis.ipynb
    ```

2. Perform squats in front of your webcam. The system will analyze the squats, display the results, and save the distance data to `squat_landmark_distances.txt`.

### Time Series Analysis
1. After recording the squat data, run the time series analysis script:
    ```bash
    python time_series_analysis.ipynb
    ```

2. The script will analyze the recorded data, visualize the results, and provide insights into the squat performance.

## Dataset
The dataset consists of distance measurements between specific body landmarks recorded during the squat exercise. The distances are saved in the file `squat_landmark_distances.txt`.

### Example Data Structure:
```
left_ankle_to_left_knee, left_ankle_to_left_hip, left_ankle_to_left_elbow, left_ankle_to_left_eye
0.123, 0.456, 0.789, 1.012
...
```

## Model Building
The model uses MediaPipe for real-time body landmark detection and OpenCV for image processing. The distance between body parts is calculated to analyze the squat performance.

### Key Components:
1. **MediaPipe Pose**: Used for real-time body landmark detection.
2. **OpenCV**: Used for handling real-time video feed and image processing.
3. **Distance Calculation**: Calculates Euclidean distances between various body landmarks to assess squat form.

## Time Series Analysis and Visualization
The time series analysis script processes the recorded data and visualizes the results. Key distances between body parts are plotted to assess squat form comprehensively.

### Key Steps:
1. **Load the Dataset**: Load the recorded distances into a pandas DataFrame.
2. **Data Cleaning**: Convert columns to numeric and handle missing values.
3. **Distance Plotting**: Plot the vertical distances between different body parts over time.

### Instructions to Interpret Results:
1. **Vertical Distance Plot**: Vertical distances between the left ankle and various body parts (hip, knee, elbow, eye) are plotted over time.
2. **Identify Patterns**: Look for patterns in the plotted data to understand the squat performance. Consistent changes in vertical distances can indicate the depth and form of squats.

### Reading the Results:
- **Time Series Plots**: Multiple plots showing the vertical distances between the left ankle and other body parts (hip, knee, elbow, eye) over time.
- **Identify Trends**: Observe the trends and patterns in the plots to evaluate squat performance. Consistent and proper squats will show regular patterns, while improper squats may show irregular patterns or anomalies.

## Project Structure
```
insect-squat-analysis/
├── real_time_squat_analysis.ipynb
├── squat_landmark_distances.txt
├── time_series_analysis.ipynb
├── README.md
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or improvements.

## Acknowledgements
- This project uses [MediaPipe](https://mediapipe.dev/) for body landmark detection.
- [OpenCV](https://opencv.org/) is used for image processing and real-time video feed handling.
- Special thanks to the contributors of the open-source libraries and tools used in this project.
