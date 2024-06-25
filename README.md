# Squat-Reps-Counting-Algorithm
This project uses pose detection data to count squat repetitions and identify incorrect reps where the person did not squat low enough. The analysis is performed using time series analysis and specific algorithms to detect peaks in the movement data.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Squat exercises are a fundamental component of many fitness routines. Accurately counting repetitions and identifying incorrect reps can help in improving workout quality and effectiveness. This project leverages time series analysis to automatically count squat reps and detect incorrect reps using pose detection output data.

## Installation
Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have the pose detection output data saved as `test_data.txt` in the project directory.
2. Run the main script to count the squat reps and visualize the results:
   ```bash
   python Squat_Rep_Counting.py
   ```

## Methodology

### Loading Pose Detection Data

The pose detection output data is loaded and parsed. The keypoints for different body parts are extracted and normalized.

```python
import pandas as pd
import matplotlib.pyplot as plt

keypoints = ['NOSE','LEFT_EYE','RIGHT_EYE','LEFT_EAR','RIGHT_EAR','LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_ELBOW','RIGHT_ELBOW','LEFT_WRIST','RIGHT_WRIST','LEFT_HIP','RIGHT_HIP','LEFT_KNEE','RIGHT_KNEE','LEFT_ANKLE','RIGHT_ANKLE']
keypoints_x, keypoints_y = [], []
for points in keypoints:
  keypoints_y.append(points + "_y")
  keypoints_x.append(points + "_x")

df_header = ['TIME'] + keypoints_y + keypoints_x
df = pd.read_csv("test_data.txt", index_col=False, names=df_header)
```

### Data Cleaning and Preprocessing

Negative entries are replaced with NaN, and NaN values are filled using the 'backfill' method to ensure data consistency.

```python
import numpy as np

df[df < 0] = np.nan
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(method='backfill', inplace=True)
```

### Time-Series Analysis

Using the `scipy.signal.find_peaks` function, peaks in the time series data are detected. Correct and incorrect squat reps are identified based on these peaks.

```python
from scipy.signal import find_peaks

def intersection(lst1, lst2):
    return [value for value in lst1 if value in lst2]

t = 75
t1 = 410
labels = ['NOSE_y','LEFT_EYE_y','RIGHT_EYE_y','LEFT_EAR_y','RIGHT_EAR_y','LEFT_SHOULDER_y','RIGHT_SHOULDER_y','LEFT_HIP_y','RIGHT_HIP_y','NOSE_x','LEFT_EYE_x', 'RIGHT_EYE_x','LEFT_EAR_x','RIGHT_EAR_x','LEFT_SHOULDER_x','RIGHT_SHOULDER_x','LEFT_HIP_x','RIGHT_HIP_x','LEFT_KNEE_x','RIGHT_KNEE_x']

thres = [80,73,73,67,61,100,100,157,157,100,157,80,73,73,67,59,151,163,157,157]
correct_squats = []

for i in range(len(labels)):
  x = df[labels[i]].to_numpy()
  peaks2, _ = find_peaks(x, prominence=25, distance=20)
  l3 = intersection(list(peaks2[peaks2 > t]), list(peaks2[peaks2 < t1]))
  total_reps = len(l3)
  peaks2, _ = find_peaks(x, prominence=25, distance=20, height=thres[i])
  l3 = intersection(list(peaks2[peaks2 > t]), list(peaks2[peaks2 < t1]))
  correct_squats.append(len(l3))

correct_squats = np.array(correct_squats)
mean = correct_squats.mean()
print('Correct Squat Reps:', int(mean))
```

### Visualization

The results are visualized using matplotlib, showing the detected peaks for correct and incorrect squat reps.

```python
plt.figure(figsize=(15,60))
for i in range(len(labels)):
  x = df[labels[i]].to_numpy()
  peaks2, _ = find_peaks(x, prominence=25, distance=20)
  l3 = intersection(list(peaks2[peaks2 > t]), list(peaks2[peaks2 < t1]))

  plt.subplot(20,2,i+1)
  peaks2, _ = find_peaks(x, prominence=25, distance=20, height=thres[i])
  l3 = intersection(list(peaks2[peaks2 > t]), list(peaks2[peaks2 < t1]))
  plt.plot(np.asarray(l3), x[np.asarray(np.unique(l3))], "xb", label='Correct Reps')
  plt.plot(x)
  plt.title(labels[i])
  plt.legend()

plt.tight_layout()
plt.show()
```

## Results

- **Correct Squat Reps:** Calculated as the mean of the detected correct reps from all keypoints.
- **Incorrect Squat Reps:** Calculated based on the difference between total detected peaks and correct peaks.

The results are displayed with the number of correct, incorrect, and total squat reps, along with visual plots indicating the detected peaks.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue to discuss any changes or improvements.
