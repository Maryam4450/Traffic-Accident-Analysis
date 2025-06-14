# 🚗 Traffic Accident Analysis & Injury Prediction

## 📝 Overview

This project is a **Streamlit-based web application** designed to analyze traffic accident data and predict the total number of injuries using a **Random Forest Regressor** model. The application provides an interactive interface for **Exploratory Data Analysis (EDA)** and real-time **injury predictions** based on user inputs for various crash-related features such as weather, road conditions, and crash type.

---

## ✨ Features

- **Exploratory Data Analysis (EDA)**:  
  - Missing value analysis
  - Summary statistics
  - Correlation heatmaps
  - Trend analysis
  - Grouped visualizations (weather, lighting, crash type, etc.)

- **Injury Prediction**:  
  Utilizes a Random Forest Regressor to predict the total number of injuries based on user-provided features.

- **Interactive Interface**:  
  - Built with Streamlit, allowing users to toggle EDA sections and input features for predictions.
  - Real-time predictions based on selected input features
  - Clean and responsive UI
  - Categorical encoding error-handling

- **Data Preprocessing**:  
  Handles missing values, encodes categorical variables, and scales numerical features for robust model performance.


## 🧠 Technologies Used

- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Handling**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`
- **Machine Learning**: `scikit-learn`
- **Deployment Ready**: Optimized with `@st.cache_data` and `@st.cache_resourc`
- **Language**: `python`
  
--------------------------------------------------------------------------------------------------------------------------------------
##  Dataset 

The application uses a dataset [traffic_accidents](https://www.kaggle.com/datasets/oktayrdeki/traffic-accidents?resource=download) from kaggle containing traffic accident records with columns such as:
- **`crash_date`**: The date the accident occurred.
- **`traffic_control_device`**: The type of traffic control device involved (e.g., traffic light, sign).
- **`weather_condition`**: The weather conditions at the time of the accident.
- **`lighting_condition`**: The lighting conditions at the time of the accident.
- **`first_crash_type`**: The initial type of the crash (e.g., head-on, rear-end).
- **`trafficway_type`**: The type of roadway involved in the accident (e.g., highway, local road).
- **`alignment`**: The alignment of the road where the accident occurred (e.g., straight, curved).
- **`roadway_surface_cond`**: The condition of the roadway surface (e.g., dry, wet, icy).
- **`road_defect`**: Any defects present on the road surface.
- **`crash_type`**: The overall type of the crash.
- **`intersection_related_i`**: Whether the accident was related to an intersection.
- **`damage`**: The extent of the damage caused by the accident.
- **`prim_contributory_cause`**: The primary cause contributing to the crash.
- **`num_units`**: The number of vehicles involved in the accident.
- **`most_severe_injury`**: The most severe injury sustained in the crash.
- **`injuries_total`**: The total number of injuries reported.
- **`injuries_fatal`**: The number of fatal injuries resulting from the accident.
- **`injuries_incapacitating`**: The number of incapacitating injuries.
- **`injuries_non_incapacitating`**: The number of non-incapacitating injuries.
- **`injuries_reported_not_evident`**: The number of injuries reported but not visibly evident.
- **`injuries_no_indication`**: The number of cases with no indication of injury.
- **`crash_hour`**: The hour the accident occurred.
- **`crash_day_of_week`**: The day of the week the accident occurred.
- **`crash_month`**: The month the accident occurred.
Note: Ensure the dataset is properly formatted and placed in the project directory before running the app.

---------------------------------------------------------------------------------
## 🤖 Model Details

- **Algorithm**: Random Forest Regressor

- **Training Split**: 
  - 70% training
  - 30% testing

### 🔄 Preprocessing Steps
- Categorical variables encoded using `LabelEncoder`
- Numerical features scaled using `StandardScaler`

### 📊 Model Evaluation
- **RMSE**: Root Mean Squared Error — Measures the average magnitude of the error.
- **R² Score**: Coefficient of Determination — Indicates how well the model explains variance in the data.

### Conclusion
The Traffic Accident Analysis and Injury Prediction application demonstrates how 
data-driven approaches can be effectively applied to enhance road safety awareness and 
support injury prevention strategies.
Through detailed exploratory analysis, the app highlights critical patterns such as peak 
accident hours, prevalent crash types, and the influence of weather and roadway conditions. 
The integrated Random Forest model offers a robust predictive capability for estimating total 
injuries based on various input features, providing a foundational step toward real-time decision 
support systems.
