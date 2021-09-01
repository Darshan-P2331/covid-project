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

## Patient record should contain the following

The record for each patient should consist of the following 

1. Pneumonia (True/False - 0/1)
2. Age (Non-negative Integer)
3. Prengancy (True/False - 0/1)
4. Diabetes (True/False - 0/1)
5. COPD (True/False - 0/1)
6. Asthma (True/False - 0/1)
7. Hypertension (True/False - 0/1)
8. Other Diseases (True/False - 0/1)
9. Cardiovascular Diseases (True/False - 0/1)
10. Obesity (True/False - 0/1)
11. Kidney Disorders (True/False - 0/1)
12. Tobacco consumer (True/False - 0/1)
13. Male (True/False - 0/1)
14. Female (True/False - 0/1)
15. Transgender (True/False - 0/1)

## Sample Program

```python
import joblib
import numpy

# Load the MLModel
MLModel = joblib.load('MLModel.sav')

patient = []

# Patient summary: 25 year-old female, obese
patient.append(numpy.array([0, 25, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0]))

# Patient summary: 80 year-old male, diabetes, cardiovascular disease
patient.append(numpy.array([0, 80, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0]))

# Prints an array with two elements
print(MLModel.predict(patient))
```
