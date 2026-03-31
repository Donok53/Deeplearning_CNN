# HW3 - CNN for Facial Expression Recognition

## Dataset

This homework uses the **FER2013** dataset.

- Train & Testing Data  
  https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data

## Download Instructions

1. Visit the Kaggle competition page above.
2. Sign in to your Kaggle account.
3. Download the dataset.
4. Extract the files into the `dataset/` directory of this project.

## Directory Structure

```text
Deeplearning_CNN/
├─ dataset/
│  └─ challenges-in-representation-learning-facial-expression-recognition-challenge/
│     └─ fer2013/
│        └─ fer2013/
│           ├─ fer2013.csv
│           ├─ fer2013.bib
│           └─ README
├─ hw3_cnn_fer2013.py
└─ README.md
```

## Execution

Run the script with:

```bash
python hw3_cnn_fer2013.py
```

If the dataset is located elsewhere, specify the file path manually:

```bash
python hw3_cnn_fer2013.py --csv <path_to_fer2013.csv>
```

## Remark

This repository does not include the FER2013 dataset file.  
Please download it separately from Kaggle.
