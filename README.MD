# Covid-19 Project

A machine learning model is used to predict whether a Covid-19 patient should be hospitalized or not based on factors such as age, gender, pre-existing comorbidities, etc.

Click **[here](https://github.com/dhruva-shashi/covid-project/blob/master/MLModel.sav?raw=true)** to download the machine learning model.

Click **[here](https://www.kaggle.com/tanmoyx/covid19-patient-precondition-dataset?select=covid.csv)** to view the Kaggle dataset.

## Factors used for analysis

- Age
- Gender
- Pre-existing comrobidities
- Pneumonia
- Pregnancy

## Pre-existing comorbidities used for analysis

- Diabetes
- Chronic obstructive pulmonary disease (COPD)
- Asthma
- Hypertension
- Cardiovascular diseases
- Kidney disorders
- Obesity
- Tobacco consumption
- Other diseases

## Usage

Use joblib to load the MLModel in your program

```python
import joblib

MLModel = joblib.load('MLModel.sav')
```

## Patient record should contain the following attributes

| No | Factor | Type |
| --- | ------ | ---- |
| 1 | Pneuomnia | True/False (1/0) |
| 2 | Age | Non-negative Integer |
| 3 | Pregnancy | True/False (1/0) |
| 4 | Diabetes | True/False (1/0) |
| 5 | COPD | True/False (1/0) |
| 6 | Asthma | True/False (1/0) |
| 7 | Hypertension | True/False (1/0) |
| 8 | Other Diseases | True/False (1/0) |
| 9 | Cardiovascular Diseases | True/False (1/0) |
| 10 | Obesity | True/False (1/0) |
| 11 | Kidney Disorders | True/False (1/0) |
| 12 | Tobacco consumer | True/False (1/0) |
| 13 | Male | True/False (1/0) |
| 14 | Female | True/False (1/0) |
| 15 | Transgender | True/False (1/0) |

## Sample Program

```python
import joblib

# Load the MLModel
MLModel = joblib.load('MLModel.sav')

patient = []

# Patient summary: 25 year-old female, obese
patient.append([0, 25, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0])

# Patient summary: 80 year-old male, diabetes, cardiovascular disease
patient.append([0, 80, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0])

# Prints an array with two elements
print(MLModel.predict(patient))
```
