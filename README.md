![Image](assets/air_quality_index.jpg)
Photo by <a href="https://unsplash.com/@mrdarkcore?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Denis Kashentsov</a> on <a href="https://unsplash.com/photos/a-person-riding-a-bike-on-a-busy-highway-g1Tab7zqg2g?utm_content=creditCopyText&utm_medium=referral&utm_source=unsplash">Unsplash</a>
      
      
      
![Python version](https://img.shields.io/badge/Python%20version-3.12-lightgrey)
![GitHub last commit](https://img.shields.io/github/last-commit/samithcsachi/Air-Quality-Health-Alert-System)
![GitHub repo size](https://img.shields.io/github/repo-size/samithcsachi/Air-Quality-Health-Alert-System)
![License](https://img.shields.io/badge/License-MIT-green)
[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)

Badge [source](https://shields.io/)

# Air-Quality-Health-Alert-System

In this project we are developing Streamlit Dashboard which alerts the residents in USA about air quality health risks.This is an end to end Machine learning portfolio project which involves the Model Development and Model training. Machine Learning Project is implemented with MLOps and CI/CD pipelines. MLOps consists of Data Ingestion, Data Validation, Data Transformation, Model Trainer, Model Evaluation,Alert Generation and Dashboard. Streamlit app is deployed in the streamlit website.

## Authors

- [Samith Chimminiyan](https://www.github.com/samithcsachi)

## Table of Contents

- [Authors](#Authors)
- [Table of Contents](#table-of-contents)
- [Problem Statement](#problem-statement)
- [Tech Stack](#tech-stack)
- [Data source](#data-source)
- [Quick glance at the results](#Quick-glance-at-the-results)
- [Lessons learned and recommendation](#lessons-learned-and-recommendation)
- [Limitation and what can be improved](#limitation-and-what-can-be-improved)
- [Work Flows](#workflows)
- [Run Locally](#run-locally)
- [Explore the notebook](#explore-the-notebook)
- [Contribution](#contribution)
- [License](#license)

## Problem Statement 

Air pollution remains a major global health concern, with millions of people exposed daily to harmful levels of pollutants. High Air Quality Index (AQI) values are directly linked to respiratory diseases, cardiovascular problems, and overall health deterioration. However, the general public often finds it difficult to interpret AQI numbers and understand their health implications, which reduces the effectiveness of air quality monitoring systems.

Traditional air quality dashboards typically provide raw AQI values without actionable context, real-time monitoring, or health-oriented alerts. This creates a gap between environmental data and informed personal decision-making.

The Air Quality Health Alert Dashboard addresses this gap by simulating air quality data, predicting AQI trends, and generating real-time health alerts and recommendations. Instead of relying on external APIs, the system uses a synthetic dataset and interactive features (e.g., sliders) to simulate AQI inputs and compare predicted vs. actual values. The dashboard provides:

- Visual comparison of predicted and actual AQI values.

- Dynamic alerts based on AQI thresholds.

- Health recommendations tailored to different AQI levels.

- Historical trend tracking for the last 10 updates.

This solution not only demonstrates how AQI prediction and health alert systems can be built end-to-end but also serves as an educational tool to raise awareness of the health risks associated with poor air quality.

## Tech Stack

- pandas
- mlflow
- notebook
- numpy
- scikit-learn
- xgboost
- seaborn
- matplotlib
- python-box
- pyYAML
-joblib
- types-pyYAML
- streamlit
- streamlit-autorefresh
- plotly