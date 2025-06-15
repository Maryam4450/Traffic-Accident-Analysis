import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("traffic_accidents.csv")
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)


    df['crash_type'] = df['crash_type'].str.split('/').str[-1].str.strip()
    df1 = df.drop(['crash_date', 'injuries_fatal', 'injuries_incapacitating', 
                   'injuries_non_incapacitating', 'injuries_reported_not_evident', 
                   'injuries_no_indication'], axis=1)

    # Encode categorical variables
    label_encoders = {}
    cat_cols = df1.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df1[col] = le.fit_transform(df1[col].astype(str))
        label_encoders[col] = le

    # Scale numerical features
    num_cols = df1.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_cols.remove('injuries_total')
    scaler = StandardScaler()
    df1[num_cols] = scaler.fit_transform(df1[num_cols])

    return df, df1, label_encoders, scaler, num_cols

# Train model
@st.cache_resource
def train_model(df1):
    X = df1.drop('injuries_total', axis=1)
    y = df1['injuries_total']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=49)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, X_train.columns.tolist(), {'rmse': rmse, 'r2': r2}

# Load data and train model
df, df1, label_encoders, scaler, num_cols = load_and_preprocess_data()
model, feature_columns, metrics = train_model(df1)

# Title and Introduction
st.markdown(
    """
    <h1 style="text-align: center; color: #06587a; font-family: 'Georgia', serif; 
    text-shadow: 2px 2px 4px #A3BFFA; border-bottom: 2px solid #FFD700; 
    padding-bottom: 5px;">🚗 Traffic Accident Analysis & Injury Prediction</h1>
    """,
    unsafe_allow_html=True
)
st.header("Introduction")
st.write("""
    This application analyzes a real-world dataset of traffic accidents to uncover patterns and 
    generate predictions. Through detailed Exploratory Data Analysis (EDA) and a Random Forest model, 
    we estimate the total number of injuries (injuries_total) based on various crash-related features such 
    as weather, road conditions, lighting, and time of the accident.
    The goal is to identify the most influential factors contributing to accident severity and enable 
    real-time prediction of potential injury impact.
""")


eda_toggle = st.sidebar.checkbox("Show EDA", value=True)
if eda_toggle:
    # EDA Section
    st.header("Exploratory Data Analysis (EDA)")
    st.subheader("Missing Value Analysis")
    missing_values = df.isnull().sum()
    missing_df = missing_values.reset_index()
    missing_df.columns = ["Feature", "Missing Values"]
    st.dataframe(missing_df)

    st.subheader("Summary Statistics")
    numerical_stats = df.select_dtypes(include=['int64', 'float64']).describe()
    st.write("Numerical Features Summary:", numerical_stats)

    # Mode for categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    mode_dict = {col: df[col].mode()[0] for col in categorical_cols}
    mode_df = pd.DataFrame(list(mode_dict.items()), columns=["Feature", "Mode"])
    st.write("### Mode for Categorical Features")
    st.dataframe(mode_df)

    st.subheader("Visualizations")
    # Correlation Matrix
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.select_dtypes(include=['int64', 'float64']).corr(), annot=True, cmap='coolwarm', fmt='.2f')
    st.pyplot(fig1)
    st.write("Correlation matrix shows strong positive correlations between injuries_total and injuries_incapacitating (0.32), non-incapacitating (0.77), and reported_not_evid (0.55); moderate negative with injuries_no_indication (-0.32, -0.25); weak ties with crash_hour, day_of_week, and month.")

    # Boxplot for Outlier Detection
    st.subheader("Outlier Detection")
    fig5, ax5 = plt.subplots(figsize=(8, 5))
    df_box = df[['injuries_total', 'injuries_fatal']].melt(var_name='Injury Type', value_name='Count')
    sns.boxplot(data=df_box, x='Injury Type', y='Count', ax=ax5)
    ax5.set_title("Outlier Detection")
    ax5.set_xlabel("Injury Type")
    ax5.set_ylabel("Count")
    st.pyplot(fig5)
    st.write("The chart shows outlier detection for injuries_total and injuries_fatal. Most injuries_total are around 0-2 with a few outliers up to 20, while injuries_fatal are mostly 0 with rare outliers around 2-3")

    # Histogram for numerical features
    st.subheader("Histogram for numerical features")
    fig2, axes2 = plt.subplots(nrows=3, ncols=3, figsize=(12, 8))
    axes2 = axes2.flatten()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for i, col in enumerate(numerical_cols[:9]):
        axes2[i].hist(df[col].dropna(), bins=30)
        axes2[i].set_title(col)
    plt.tight_layout()
    st.pyplot(fig2)
    st.write("This shows most crashes involve 2 units, have 0 injuries (total, fatal, incapacitating, non-incapacitating, reported but not evident), and occur between 3-5 PM, with a slight peak on days 5, 6, 7")


    # Crash Hour Distribution
    st.subheader('Crash Hour Distribution')
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    df['crash_hour'].hist(bins=24, ax=ax3)
    ax3.set_title("Crashes by Hour of Day")
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("Frequency")
    st.pyplot(fig3)

    # Grouped Aggregations
    st.subheader('Plot average injuries by weather condition')
    fig8, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df.groupby('weather_condition')['injuries_total'].mean().sort_values().index, 
            df.groupby('weather_condition')['injuries_total'].mean().sort_values().values)
    plt.title("Avg Injuries by Weather Condition")
    plt.xlabel("Average Injuries")
    plt.ylabel("Weather Condition")
    plt.tick_params()
    ax.bar_label(ax.containers[0], fmt='%.2f')
    st.pyplot(fig8)

    # Create pairplot for injuries_total, injuries_fatal, and num_units
    st.subheader('Pairplot for injuries_total, injuries_fatal, and num_units')
    fig = sns.pairplot(df[['injuries_total', 'injuries_fatal', 'num_units']]).figure
    st.pyplot(fig)

    # Trend Analysis
    df_temp = df.copy()
    df_temp['crash_date'] = pd.to_datetime(df_temp['crash_date'], errors='coerce')
    st.subheader("Trend Analysis")
    fig7, ax7 = plt.subplots(figsize=(10, 6))
    ax7.plot(df_temp['crash_date'].dt.date.value_counts().sort_index())
    ax7.set_title("Crashes Over Time")
    ax7.set_xlabel("Date")
    ax7.set_ylabel("Number of Crashes")
    st.pyplot(fig7)
    st.write("The chart shows crashes over time from 2014 to 2024. Crash numbers were low until 2017, then rose sharply, peaking around 2018-2022 with occasional spikes up to 140, and slightly declined by 2024.")

    st.subheader("Relevant Analysis")
    # Pie Chart for Top 5 Crash Types
    fig10, ax10 = plt.subplots(figsize=(4, 6))
    ax10.pie(df['first_crash_type'].value_counts().head(5), labels=df['first_crash_type'].value_counts().head(5).index, autopct='%1.1f%%')
    ax10.set_title("Top 5 Crash Types in Accidents")
    st.pyplot(fig10)
    st.write("The pie chart shows the top 5 crash types in accidents: Turning (34.2%), Angle (27.9%), Rear End (22.4%), Sideswipe Same (10.7%), and Pedestrian (4.8%)")

    # AM/PM Stacked Bar Charts
    st.subheader("Comparison between AM/PM Accident by Hours and Trafficway Type")
    x_axis = 'crash_hour'
    stack_by = 'trafficway_type'
    top_categories = df[stack_by].value_counts().index[:5]
    df_filtered = df[df[stack_by].isin(top_categories)]
    df_am = df_filtered[df_filtered[x_axis] < 12]
    df_pm = df_filtered[df_filtered[x_axis] >= 12]
    pivot_am = df_am.pivot_table(index=x_axis, columns=stack_by, aggfunc='size', fill_value=0)
    pivot_pm = df_pm.pivot_table(index=x_axis, columns=stack_by, aggfunc='size', fill_value=0)

    fig4, (ax4, ax5) = plt.subplots(1, 2, figsize=(20, 7))
    pivot_am.plot(kind='bar', stacked=True, cmap='Blues', ax=ax4)
    ax4.set_title('AM Accidents by Hour and Trafficway Type')
    ax4.set_xlabel('Hour (0–11)')
    ax4.set_ylabel('Accident Count')
    ax4.tick_params(axis='x', rotation=0)
    pivot_pm.plot(kind='bar', stacked=True, cmap='Oranges', ax=ax5)
    ax5.set_title('PM Accidents by Hour and Trafficway Type')
    ax5.set_xlabel('Hour (12–23)')
    ax5.set_ylabel('Accident Count')
    ax5.tick_params(axis='x', rotation=0)
    st.pyplot(fig4)
    st.write("This shows that traffic accidents peak during rush hours—around 8 AM and between 3–6 PM—with the highest counts occurring on non-divided and four-way roads. AM accidents gradually rise from 6 AM, while PM accidents are consistently high from noon to evening, indicating increased risk during commuting hours.")
    
    #Crash Hour vs Total Injuries
    st.subheader("Crash Hour vs Total Injuries")
    fig11, ax11 = plt.subplots(figsize=(8, 5))
    ax11.scatter(df['crash_hour'], df['injuries_total'], color='orange', alpha=0.6)
    ax11.set_xlabel('Crash Hour')
    ax11.set_ylabel('Total Injuries')
    ax11.set_title('Scatter Plot: Crash Hour vs Total Injuries')
    ax11.grid(True)
    st.pyplot(fig11)
    st.write("The scatter plot shows total injuries by crash hour. Most injuries (0-5) occur across all hours, with a peak around 10-15 hours. Higher injuries (10-20) are rare and scattered.")

    #Average Injuries by Lighting Condition
    st.subheader("Average Injuries by Lighting Condition")
    fig12, ax12 = plt.subplots(figsize=(10, 6))
    avg_injuries_by_light = df.groupby('lighting_condition')['injuries_total'].mean().sort_values()
    avg_injuries_by_light.plot(kind='barh', color='skyblue', ax=ax12)
    ax12.set_xlabel("Average Injuries")
    ax12.set_title("Average Injuries by Lighting Condition")
    ax12.grid(axis='x', linestyle='--', alpha=0.7)
    st.pyplot(fig12)
    st.write("This shows average injuries by lighting condition. Darkness (lit/unlit) and dawn/dusk have the highest averages (~0.35-0.4), daylight is lower (~0.3), and unknown is the lowest (~0.1).")

    #Distribution of Injuries by Weather Condition (Violin Plot)
    st.subheader("Distribution of Injuries by Weather Condition")
    fig13, ax13 = plt.subplots(figsize=(12, 6))
    sns.violinplot(x='weather_condition', y='injuries_total', data=df, inner='quartile', palette='Set3', ax=ax13)
    ax13.set_xticklabels(ax13.get_xticklabels(), rotation=45)
    ax13.set_title("Distribution of Injuries by Weather Condition (Violin Plot)")
    ax13.set_xlabel("Weather Condition")
    ax13.set_ylabel("Injuries Total")
    st.pyplot(fig13)
    st.write("This shows injury distribution by weather condition. Clear and rain have the highest injury counts (up to 20), while snow, cloudy/overcast, unknown, fog/smoke/haze, and others range lower (0-10), with minimal injuries in severe crosswind, sleet/hail, and blowing sand/soil")

ml_toggle = st.sidebar.checkbox("Show Machine Learning Model", value=True)
if ml_toggle:
    # Model Section
    st.header("Machine Learning Model")
    st.subheader("Model Description")
    st.write("""
    A Random Forest Regressor model is  used to predict the total number of injuries (`injuries_total`) based on all available features. The model was trained on 70% of the data and evaluated on 30% test data.
    """) 

    # Display model performance metrics
    st.subheader("Model Performance")
    st.write(f"RMSE: {metrics['rmse']:.2f}")
    st.write(f"R2 Score: {metrics['r2']:.2f}")

    st.subheader("Runtime Prediction")
    st.write("Enter the following features to predict injuries:")

    # User input widgets for all key features
    crash_hour = st.slider("Crash Hour (0-23)", 0, 23, 12)
    traffic_control_device = st.selectbox("Traffic Control Device", df['traffic_control_device'].value_counts().index[:8])
    damage = st.selectbox("Damage", df['damage'].value_counts().index[:8])
    crash_day_of_week = st.selectbox("Crash Day of Week", df['crash_day_of_week'].value_counts().index[:8])
    intersection_related_i = st.selectbox("Intersection Related", df['intersection_related_i'].value_counts().index[:8])
    trafficway_type = st.selectbox("Trafficway Type", df['trafficway_type'].value_counts().index[:5])
    weather_condition = st.selectbox("Weather Condition", df['weather_condition'].value_counts().index[:5])
    first_crash_type = st.selectbox("First Crash Type", df['first_crash_type'].value_counts().index[:5])
    lighting_condition = st.selectbox("Lighting Condition", df['lighting_condition'].value_counts().index[:5])
    roadway_surface_cond = st.selectbox("Roadway Surface Condition", df['roadway_surface_cond'].value_counts().index[:5])
    alignment = st.selectbox("Alignment", df['alignment'].value_counts().index[:5])
    crash_type = st.selectbox("Crash Type ", df['crash_type'].value_counts().index[:5])
    road_defect = st.selectbox("Road Defect", df['road_defect'].value_counts().index[:5])
    prim_contributory_cause = st.selectbox("Primary Contributory Cause", df['prim_contributory_cause'].value_counts().index[:5])
    most_severe_injury = st.selectbox("Most Severe Injury", df['most_severe_injury'].value_counts().index[:5])
    crash_month = st.selectbox("Crash Month", df['crash_month'].value_counts().index[:12])
    num_units = st.slider("Number of Units", 1, 10, 1)

    # Prepare input_data with all features used in training
    input_data = pd.DataFrame(index=[0], columns=feature_columns)
    input_data['crash_hour'] = crash_hour
    input_data['traffic_control_device'] = traffic_control_device
    input_data['trafficway_type'] = trafficway_type
    input_data['weather_condition'] = weather_condition
    input_data['first_crash_type'] = first_crash_type
    input_data['lighting_condition'] = lighting_condition
    input_data['roadway_surface_cond'] = roadway_surface_cond
    input_data['alignment'] = alignment
    input_data['crash_type'] = crash_type
    input_data['road_defect'] = road_defect
    input_data['prim_contributory_cause'] = prim_contributory_cause
    input_data['most_severe_injury'] = most_severe_injury
    input_data['crash_month'] = crash_month
    input_data['num_units'] = num_units
    input_data['damage'] = damage
    input_data['crash_day_of_week'] = crash_day_of_week
    input_data['intersection_related_i'] = intersection_related_i

    # Fill any remaining missing values with defaults
    for col in feature_columns:
        if pd.isna(input_data[col].iloc[0]):
            if col in df.select_dtypes(include=['object']).columns:
                input_data[col] = df[col].mode()[0]
            elif col in df.select_dtypes(include=['int64', 'float64']).columns and col != 'injuries_total':
                input_data[col] = 0

    # Encode categorical variables using saved encoders
    cat_cols = [col for col in feature_columns if col in label_encoders]
    for col in cat_cols:
        try:
            input_data[col] = label_encoders[col].transform(input_data[col].astype(str))
        except ValueError:
            st.error(f"Unknown category in `{col}`. Please select a known category.")
            st.stop()

    # Scale all features
    input_data_array = scaler.transform(input_data)

    # Predict using the scaled array
    if st.button("Predict Injuries"):
        try:
            prediction = model.predict(input_data_array)
            if prediction[0] < 0:
                st.success(f"Predicted Total Injuries: 0")
            else:
                st.success(f"Predicted Total Injuries: {round(prediction[0])}")
        except ValueError as e:
            st.error(f"Prediction failed: {str(e)}")
        except KeyError as e:
            st.error(f"Encoding failed: {str(e)}")

# Conclusion
st.header("Conclusion")
st.write("""
The Traffic Accident Analysis and Injury Prediction application demonstrates how 
data-driven approaches can be effectively applied to enhance road safety awareness and 
support injury prevention strategies.
Through detailed exploratory analysis, the app highlights critical patterns such as peak 
accident hours, prevalent crash types, and the influence of weather and roadway conditions. 
The integrated Random Forest model enhances the app with real-time injury prediction based 
on crash features, offering a practical foundation for decision support systems in traffic 
safety and emergency response.
""")
