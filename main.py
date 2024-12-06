import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle


def create_flight_app():
    st.title("✈️ Flight Analysis Dashboard")

    data = './data/flights.csv'
    df = pd.read_csv(data)

    dest_mapping = {
        'ATL': 'Atlanta, GA',
        'BOS': 'Boston, MA',
        'DFW': 'Dallas/Fort Worth, TX',
        'LAX': 'Los Angeles, CA',
        'ORD': 'Chicago, IL',
        'SFO': 'San Francisco, CA',
        'SEA': 'Seattle, WA',
        'MSP': 'Minneapolis, MN',
        'EWR': 'Newark, NJ',
        'DEN': 'Denver, CO',
        'IAH': 'Houston, TX',
        'BDL': 'Hartford, CT',
        'DCA': 'Washington, DC',
        'FAI': 'Fairbanks, AK',
        'OKC': 'Oklahoma City, OK',
        'MSY': 'New Orleans, LA'
    }

    df['DEST_FULL'] = df['DEST'].map(dest_mapping)
    df['ORIGIN_FULL'] = df['ORIGIN'].map(dest_mapping)

    df['Status'] = pd.cut(df['ARR_DELAY'],
                          bins=[-float('inf'), -15, 15, float('inf')],
                          labels=['Early', 'On-time', 'Delayed'])

    st.header("Select Analysis View")
    col1, col2 = st.columns(2)
    with col1:
        analysis_type = st.radio(
            "Analysis Type",
            ["By Destination", "By Airline"]
        )
    with col2:
        view_type = st.radio(
            "View Type",
            ["Overall", "By Source Airport"]
        )

    if analysis_type == "By Destination":

        dest_options = df['DEST_FULL'].dropna().unique()
        selected_dest_full = st.selectbox(
            "Select Destination", sorted(dest_options))
        selected_dest = df[df['DEST_FULL'] ==
                           selected_dest_full]['DEST'].iloc[0]

        if view_type == "Overall":
            dest_data = df[df['DEST'] == selected_dest]
        else:
            selected_origin_full = st.selectbox("Select Source Airport",
                                                sorted(df['ORIGIN_FULL'].dropna().unique()))
            selected_origin = df[df['ORIGIN_FULL'] ==
                                 selected_origin_full]['ORIGIN'].iloc[0]
            dest_data = df[(df['DEST'] == selected_dest) &
                           (df['ORIGIN'] == selected_origin)]

        st.header("Flight Statistics")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            total_flights = len(dest_data)
            st.metric("Total Flights", f"{total_flights:,}")
        with metric_col2:
            avg_delay = dest_data['ARR_DELAY'].mean()
            st.metric("Average Delay", f"{avg_delay:.1f} mins")
        with metric_col3:
            on_time_rate = (dest_data['Status'].isin(
                ['On-time', 'Early'])).mean() * 100
            st.metric("On-Time Rate", f"{on_time_rate:.1f}%")
        with metric_col4:
            early_rate = (dest_data['Status'] == 'Early').mean() * 100
            st.metric("Early Rate", f"{early_rate:.1f}%")

        col1, col2 = st.columns(2)

        with col1:
            status_counts = dest_data['Status'].value_counts()
            fig1 = px.pie(values=status_counts.values,
                          names=status_counts.index,
                          title=f"Flight Status Distribution for {selected_dest_full}",
                          color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig1)

        with col2:
            avg_delays = dest_data.groupby(
                'AIRLINE')['ARR_DELAY'].mean().reset_index()
            fig2 = px.bar(avg_delays,
                          x='AIRLINE',
                          y='ARR_DELAY',
                          title=f"Average Delays by Carrier to {selected_dest_full}")
            st.plotly_chart(fig2)

        st.subheader("Delay Analysis")
        delay_cols = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
                      'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']

        delay_data = dest_data[delay_cols].sum()
        fig3 = px.bar(x=delay_data.index,
                      y=delay_data.values,
                      title=f"Delay Reasons for {selected_dest_full}")
        st.plotly_chart(fig3)

    else:
        selected_airline = st.selectbox(
            "Select Airline", df['AIRLINE'].unique())

        if view_type == "Overall":
            airline_data = df[df['AIRLINE'] == selected_airline]
        else:
            selected_origin_full = st.selectbox("Select Source Airport",
                                                sorted(df['ORIGIN_FULL'].dropna().unique()))
            selected_origin = df[df['ORIGIN_FULL'] ==
                                 selected_origin_full]['ORIGIN'].iloc[0]
            airline_data = df[(df['AIRLINE'] == selected_airline) &
                              (df['ORIGIN'] == selected_origin)]

        st.header("Airline Performance")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            total_flights = len(airline_data)
            st.metric("Total Flights", f"{total_flights:,}")
        with metric_col2:
            avg_delay = airline_data['ARR_DELAY'].mean()
            st.metric("Average Delay", f"{avg_delay:.1f} mins")
        with metric_col3:
            on_time_rate = (airline_data['Status'].isin(
                ['On-time', 'Early'])).mean() * 100
            st.metric("On-Time Rate", f"{on_time_rate:.1f}%")
        with metric_col4:
            avg_distance = airline_data['DISTANCE'].mean()
            st.metric("Avg Distance", f"{avg_distance:.0f} miles")

        col1, col2 = st.columns(2)

        with col1:
            status_counts = airline_data['Status'].value_counts()
            fig1 = px.pie(values=status_counts.values,
                          names=status_counts.index,
                          title=f"Flight Status Distribution for {selected_airline}",
                          color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig1)

        with col2:
            dest_delays = airline_data.groupby(
                'DEST_FULL')['ARR_DELAY'].mean().reset_index()
            fig2 = px.bar(dest_delays,
                          x='DEST_FULL',
                          y='ARR_DELAY',
                          title=f"Average Delays by Destination")
            fig2.update_layout(xaxis_title="Destination",
                               yaxis_title="Average Delay (minutes)")
            st.plotly_chart(fig2)

        st.subheader("Detailed Analysis")

        col3, col4, col5 = st.columns(3)
        with col3:
            avg_taxi_out = airline_data['TAXI_OUT'].mean()
            st.metric("Avg Taxi Out", f"{avg_taxi_out:.1f} mins")
        with col4:
            avg_taxi_in = airline_data['TAXI_IN'].mean()
            st.metric("Avg Taxi In", f"{avg_taxi_in:.1f} mins")
        with col5:
            avg_air_time = airline_data['AIR_TIME'].mean()
            st.metric("Avg Air Time", f"{avg_air_time:.1f} mins")


def create_prediction_page():
    st.title("✈️ Flight Delay Prediction")
    
    try:
        with open('random_forest_delay_model.pkl', 'rb') as file:
            model_data = pickle.load(file)
            
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        
        st.header("Enter Flight Details")
        
        col1, col2 = st.columns(2)
        with col1:
            dep_delay = st.number_input("Departure Delay (minutes)", 
                                      min_value=-60, 
                                      max_value=300, 
                                      value=0)
            distance = st.number_input("Flight Distance (miles)", 
                                     min_value=100, 
                                     max_value=3000, 
                                     value=1000)
            dep_hour = st.number_input("Departure Hour (0-23)", 
                                     min_value=0, 
                                     max_value=23, 
                                     value=14)
            
        with col2:
            cancelled = st.selectbox("Flight Cancelled", 
                                   options=[0, 1], 
                                   format_func=lambda x: "No" if x == 0 else "Yes")
            diverted = st.selectbox("Flight Diverted", 
                                  options=[0, 1], 
                                  format_func=lambda x: "No" if x == 0 else "Yes")
            month = st.slider("Month", 1, 12, 7)
            day = st.slider("Day", 1, 31, 15)

        
        day_of_week = pd.Timestamp(2024, month, day).dayofweek
        is_weekend = int(day_of_week >= 5)
        is_morning = int(6 <= dep_hour <= 12)
        is_evening = int(dep_hour >= 18)
        is_summer = int(month in [6, 7, 8])
        is_winter = int(month in [12, 1, 2])

        '''
        \nTraining Accuracy: 0.8489
        \nTesting Accuracy: 0.8468
        '''

        if st.button("Predict Delay Probability"):
            if cancelled == 1:
                
                st.header("Prediction Results")
                st.metric("Status", "Cancelled")
                
            else:
                
                input_data = {
                    'DEP_DELAY': dep_delay,
                    'DISTANCE': distance,
                    'CANCELLED': cancelled,
                    'DIVERTED': diverted,
                    'DEP_HOUR': dep_hour,
                    'DAY_OF_WEEK': day_of_week,
                    'IS_WEEKEND': is_weekend,
                    'IS_MORNING': is_morning,
                    'IS_EVENING': is_evening,
                    'IS_SUMMER': is_summer,
                    'IS_WINTER': is_winter,
                    'MONTH': month,
                    'DAY': day
                }
                
                
                input_df = pd.DataFrame([input_data])[features]
                input_scaled = scaler.transform(input_df)
                
                
                delay_probability = model.predict_proba(input_scaled)[0][1]
                
                
                if diverted == 1:
                    delay_probability = min(delay_probability + 0.4, 1.0)
                
                is_delayed = delay_probability > 0.5
                
                
                st.header("Prediction Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Delay Probability", f"{delay_probability:.1%}")
                with col2:
                    status = "Likely Delayed" if is_delayed else "Likely On Time"
                    st.metric("Predicted Status", status)
                
                
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = float(delay_probability * 100),
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Delay Probability"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 33], 'color': "lightgreen"},
                            {'range': [33, 66], 'color': "yellow"},
                            {'range': [66, 100], 'color': "salmon"}
                        ]
                    }
                ))
                
                st.plotly_chart(fig)
            
    except FileNotFoundError:
        st.error("Model file not found. Please ensure the model is trained and saved first.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page", ["Flight Analysis", "Delay Prediction"])

    if page == "Flight Analysis":
        create_flight_app()
    else:
        create_prediction_page()


if __name__ == "__main__":
    main()