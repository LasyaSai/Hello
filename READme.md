# Flight Analysis Dashboard Documentation

## Overview
The Flight Analysis Dashboard is a real-time analytics platform built with Streamlit for analyzing flight performance metrics and delays.

## Technical Stack
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
```

## Core Components
```python
def create_flight_app():
    st.title("✈️ Flight Analysis Dashboard")
    data = './data/flights.csv'
    df = pd.read_csv(data)
```

## Data Processing

### Airport Mapping
```python
dest_mapping = {
    'ATL': 'Atlanta, GA',
    'BOS': 'Boston, MA',
    'DFW': 'Dallas/Fort Worth, TX'
}
df['DEST_FULL'] = df['DEST'].map(dest_mapping)
df['ORIGIN_FULL'] = df['ORIGIN'].map(dest_mapping)
```

### Flight Classification
```python
df['Status'] = pd.cut(df['ARR_DELAY'],
                      bins=[-float('inf'), -15, 15, float('inf')],
                      labels=['Early', 'On-time', 'Delayed'])
```

Status Classification Model:
$$
Status = \begin{cases} 
Early & \text{if } \delta < -15 \\
On\text{-}time & \text{if } -15 \leq \delta \leq 15 \\
Delayed & \text{if } \delta > 15
\end{cases}
$$

## Statistical Models

### Delay Calculations
Average Delay:
$$\bar{D} = \frac{1}{n}\sum_{i=1}^{n} d_i$$

On-Time Performance Rate:
$$OTP = \frac{|\{f \in F : -15 \leq delay(f) \leq 15\}|}{|F|} \times 100\%$$

### Flight Time Components
Total Flight Time:
$$T_{total} = T_{taxi\_out} + T_{air} + T_{taxi\_in}$$

```python
avg_taxi_out = airline_data['TAXI_OUT'].mean()
avg_taxi_in = airline_data['TAXI_IN'].mean()
avg_air_time = airline_data['AIR_TIME'].mean()
```

### Distance Analysis
Average Distance:
$$\bar{d} = \frac{\sum_{i=1}^{n} DISTANCE_i}{n}$$

## Data Analysis Components

### Filtering Logic
```python
if view_type == "Overall":
    dest_data = df[df['DEST'] == selected_dest]
else:
    dest_data = df[(df['DEST'] == selected_dest) & 
                   (df['ORIGIN'] == selected_origin)]
```

### Visualization
```python
def plot_time_series(data, metric):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data.index, y=data[metric], mode='lines')
    )
    return fig
```

Time Series Decomposition:
$$Y_t = T_t + S_t + R_t$$

### Performance Metrics
```python
def aggregate_metrics(df, group_by, metrics):
    return df.groupby(group_by)[metrics].agg({
        'mean': 'mean',
        'std': 'std',
        'count': 'count'
    })
```

Standard Deviation:
$$\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

## System Architecture

### Data Pipeline
```python
class FlightDataPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load_data(self):
        return pd.read_csv(self.data_path)
        
    def process_data(self, df):
        df = self.clean_data(df)
        df = self.add_derived_features(df)
        return df
```

### Caching
```python
@st.cache_data
def load_flight_data():
    return pd.read_csv('./data/flights.csv')
```

Caching Logic:
```math
C(f(x)) = \begin{cases} 
cached\_result & \text{if } hash(x) \text{ exists} \\
f(x) & \text{otherwise}
\end{cases}
```
## Performance Monitoring
Response Time:
$$Response Time = T_{end} - T_{start}$$

Memory Usage:
$$Memory Usage = \frac{M_{used}}{M_{total}}$$

## Configuration
```python
def get_config():
    return {
        'data_path': os.getenv('FLIGHT_DATA_PATH', './data/flights.csv'),
        'cache_ttl': int(os.getenv('CACHE_TTL', 3600)),
        'debug_mode': bool(os.getenv('DEBUG_MODE', False))
    }
```
