import json
import numpy as np
import pandas as pd
import streamlit as st
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import run_deployment_pipeline

def main():

    st.title("Food Delivery Time Prediction")

    # Adjust spacing between columns
    col1, _,col2 = st.columns([0.7, 0.1, 1.3], gap="medium")

    with col1:
        distance_km = st.slider("**Distance (km)**", 0.5, 30.0, step=0.1)

        preparation_time_min = st.slider("**Preparation Time (min)**", 0, 30, step=1)

        courier_experience_yrs = st.slider("**Courier Experience (yrs)**", 0, 10, step=1)
    
    with col2:

        weather = st.radio("**Weather**", ["Foggy", "Rainy", "Snowy", "Windy", "Clear"], horizontal=True)

        traffic_level = st.radio("**Traffic Level**", ["Low", "Medium", "High"], horizontal=True)

        time_of_day = st.radio("**Time of Day**", ["Morning", "Afternoon", "Evening", "Night"], horizontal=True)

        vehicle_type = st.radio("**Vehicle Type**", ["Car", "Scooter", "Bike"], horizontal=True)

    weather_foggy = 0
    weather_snowy = 0
    weather_rainy = 0
    weather_windy = 0
    
    if weather == "Foggy":
        weather_foggy = 1
    elif weather == "Rainy":
        weather_rainy = 1
    elif weather == "Snowy":
        weather_snowy = 1
    elif weather == "Windy":
        weather_windy = 1

    traffic_level_low = 0
    traffic_level_medium = 0
    
    if traffic_level == "Low":
        traffic_level_low = 1
    elif traffic_level == "Medium":
        traffic_level_medium = 1

    time_of_day_morning = 0
    time_of_day_afternoon = 0
    time_of_day_evening = 0
    time_of_day_night = 0
    
    if time_of_day == "Morning":
        time_of_day_morning = 1
    elif time_of_day == "Afternoon":
        time_of_day_afternoon = 1
    elif time_of_day == "Evening":
        time_of_day_evening = 1
    elif time_of_day == "Night":
        time_of_day_night = 1

    vehicle_type_car = 0
    vehicle_type_scooter = 0
    
    if vehicle_type == "Car":
        vehicle_type_car = 1
    elif vehicle_type == "Scooter":
        vehicle_type_scooter = 1

    if st.button("Predict"):
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            run_deployment_pipeline(config="deploy")

        df = pd.DataFrame(
            {
                "Distance_km": [distance_km],
                "Preparation_Time_min": [preparation_time_min],
                "Courier_Experience_yrs": [courier_experience_yrs],
                "Weather_Foggy": [weather_foggy],
                "Weather_Rainy": [weather_rainy],
                "Weather_Snowy": [weather_snowy],
                "Weather_Windy": [weather_windy],
                "Traffic_Level_Low": [traffic_level_low],
                "Traffic_Level_Medium": [traffic_level_medium],
                "Time_of_Day_Morning": [time_of_day_morning],
                "Time_of_Day_Afternoon": [time_of_day_afternoon],
                "Time_of_Day_Evening": [time_of_day_evening],
                "Time_of_Day_Night": [time_of_day_night],
                "Vehicle_Type_Car": [vehicle_type_car],
                "Vehicle_Type_Scooter": [vehicle_type_scooter],
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)
        
        prediction_in_minutes = round(pred[0][0], 2)
        hours = int(prediction_in_minutes // 60)
        minutes = int(prediction_in_minutes % 60)
        seconds = int((prediction_in_minutes * 60) % 60)

        if hours > 0:
            formatted_time = f"{hours}h {minutes}min {seconds}s"
        else:
            formatted_time = f"{minutes}min {seconds}s"

        st.success(
            f"The predicted food delivery time is: {formatted_time}"
        )

if __name__ == "__main__":
    main()
