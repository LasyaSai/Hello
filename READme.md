# Flight Analysis Dashboard Technical Documentation

## Table of Contents
1. [Overview](#overview)
2. [Technical Architecture](#technical-architecture)
3. [Code Components](#code-components)
4. [Mathematical Models](#mathematical-models)
5. [Implementation Details](#implementation-details)

## Overview

The Flight Analysis Dashboard is a real-time analytics platform built with Streamlit for analyzing flight performance metrics and delays.

## Technical Architecture

### Dependencies
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
```

### Core Components
1. Data Processing Engine
2. Interactive UI
3. Statistical Analysis Module
4. Visualization System

## Code Components

### Data Loading and Preprocessing

```python
def create_flight_app():
    st.title("✈️ Flight Analysis Dashboard")
    
    # Data loading
    data = './data/flights.csv'
    df = pd.read_csv(data)
```

The application initializes by loading flight data from a CSV file. The data structure includes:
- Flight identifiers
- Temporal information
- Delay metrics
- Geographic data

### Airport Mapping System

```python
dest_mapping = {
    'ATL': 'Atlanta, GA',
    'BOS': 'Boston, MA',
    'DFW': 'Dallas/Fort Worth, TX',
    # ... additional mappings
}

df['DEST_FULL'] = df['DEST'].map(dest_mapping)
df['ORIGIN_FULL'] = df['ORIGIN'].map(dest_mapping)
```

This mapping system converts IATA codes to human-readable airport names using a dictionary lookup.

### Flight Status Classification

```python
df['Status'] = pd.cut(df['ARR_DELAY'],
                      bins=[-float('inf'), -15, 15, float('inf')],
                      labels=['Early', 'On-time', 'Delayed'])
```

The status classification follows this mathematical model:

\[ 
\text{Status} = \begin{cases} 
\text{Early} & \text{if } \delta < -15 \\
\text{On-time} & \text{if } -15 \leq \delta \leq 15 \\
\text{Delayed} & \text{if } \delta > 15
\end{cases}
\]

Where \(\delta\) represents the arrival delay in minutes.

## Mathematical Models

### Delay Statistics

1. **Average Delay Calculation**
\[ \bar{D} = \frac{1}{n}\sum_{i=1}^{n} d_i \]
where:
- \(\bar{D}\) is the mean delay
- \(d_i\) is the delay of flight i
- \(n\) is the total number of flights

```python
avg_delay = dest_data['ARR_DELAY'].mean()
```

2. **On-Time Performance Rate**
\[ \text{OTP} = \frac{|\{f \in F : -15 \leq \text{delay}(f) \leq 15\}|}{|F|} \times 100\% \]
where F is the set of all flights

```python
on_time_rate = (dest_data['Status'].isin(['On-time','Early'])).mean() * 100
```

### Performance Metrics

1. **Total Flight Time**
\[ T_{\text{total}} = T_{\text{taxi\_out}} + T_{\text{air}} + T_{\text{taxi\_in}} \]

```python
avg_taxi_out = airline_data['TAXI_OUT'].mean()
avg_taxi_in = airline_data['TAXI_IN'].mean()
avg_air_time = airline_data['AIR_TIME'].mean()
```

2. **Distance Analysis**
\[ \bar{d} = \frac{\sum_{i=1}^{n} \text{DISTANCE}_i}{n} \]

```python
avg_distance = airline_data['DISTANCE'].mean()
```

## Implementation Details

### Interactive UI Components

```python
st.header("Select Analysis View")
col1, col2 = st.columns(2)
with col1:
    analysis_type = st.radio(
        "Analysis Type",
        ["By Destination", "By Airline"]
    )
```

### Data Filtering Logic

```python
if view_type == "Overall":
    dest_data = df[df['DEST'] == selected_dest]
else:
    dest_data = df[(df['DEST'] == selected_dest) & 
                   (df['ORIGIN'] == selected_origin)]
```

This implements Boolean indexing with the following logical operation:
\[ \text{filtered\_data} = \{f \in F : \text{dest}(f) = d \land \text{origin}(f) = o\} \]

### Visualization Components

1. **Status Distribution**
```python
status_counts = dest_data['Status'].value_counts()
fig1 = px.pie(values=status_counts.values,
              names=status_counts.index,
              title=f"Flight Status Distribution")
```

2. **Delay Analysis**
```python
avg_delays = dest_data.groupby('AIRLINE')['ARR_DELAY'].mean()
fig2 = px.bar(avg_delays,
              x='AIRLINE',
              y='ARR_DELAY',
              title="Average Delays by Carrier")
```

The delay distribution follows:
\[ P(\text{delay}) = \frac{\text{count of delays in bin}}{\text{total flights}} \]

### Performance Optimization

1. **Memory Management**
```python
# Efficient boolean indexing
dest_data = df[df['DEST'] == selected_dest]
```

2. **Computational Efficiency**
The groupby operations use the following optimization:
\[ \text{group\_mean} = \frac{\sum_{i \in G} x_i}{|G|} \]
where G is the group of flights sharing a common attribute.

## Statistical Analysis Components

### Delay Reason Analysis

```python
delay_cols = ['DELAY_DUE_CARRIER', 'DELAY_DUE_WEATHER', 'DELAY_DUE_NAS',
              'DELAY_DUE_SECURITY', 'DELAY_DUE_LATE_AIRCRAFT']

delay_data = dest_data[delay_cols].sum()
```

The total delay for each category follows:
\[ D_{\text{total}} = \sum_{i=1}^{n} (D_{\text{carrier}} + D_{\text{weather}} + D_{\text{nas}} + D_{\text{security}} + D_{\text{late}}) \]

### Metric Calculations

```python
metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
with metric_col1:
    total_flights = len(dest_data)
    st.metric("Total Flights", f"{total_flights:,}")
```

#### Performance Ratios

1. **Early Flight Ratio**
\[ R_{\text{early}} = \frac{|\{f \in F : \text{delay}(f) < -15\}|}{|F|} \times 100\% \]

```python
early_rate = (dest_data['Status'] == 'Early').mean() * 100
```

2. **Delay Distribution**
\[ P(D_{\text{type}}) = \frac{\sum D_{\text{type}}}{\sum D_{\text{total}}} \]

## Advanced Visualization Components

### Time Series Analysis

```python
def plot_time_series(data, metric):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=data.index, y=data[metric], mode='lines')
    )
    return fig
```

The time series decomposition follows:
\[ Y_t = T_t + S_t + R_t \]
where:
- \(T_t\) is the trend component
- \(S_t\) is the seasonal component
- \(R_t\) is the residual component

### Performance Heatmap

```python
def create_heatmap(data, x, y, values):
    pivot_table = data.pivot_table(
        values=values,
        index=y,
        columns=x,
        aggfunc='mean'
    )
    return px.imshow(pivot_table)
```

The heatmap intensity is calculated as:
\[ H_{i,j} = \frac{\sum_{k \in C_{i,j}} v_k}{|C_{i,j}|} \]
where \(C_{i,j}\) is the set of flights for cell (i,j)

## System Architecture Components

### Data Flow Pipeline

```python
class FlightDataPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        
    def load_data(self):
        return pd.read_csv(self.data_path)
        
    def process_data(self, df):
        # Data cleaning and preprocessing
        df = self.clean_data(df)
        df = self.add_derived_features(df)
        return df
```

### Caching Mechanism

```python
@st.cache_data
def load_flight_data():
    return pd.read_csv('./data/flights.csv')
```

The caching follows the principle:
\[ C(f(x)) = \begin{cases} 
\text{cached\_result} & \text{if } hash(x) \text{ exists} \\
f(x) & \text{otherwise}
\end{cases} \]

## Performance Optimization Techniques

### Data Aggregation

```python
def aggregate_metrics(df, group_by, metrics):
    return df.groupby(group_by)[metrics].agg({
        'mean': 'mean',
        'std': 'std',
        'count': 'count'
    })
```

The standard deviation calculation:
\[ \sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2} \]

### Memory Management

```python
def optimize_dataframe(df):
    for col in df.select_dtypes(include=['float64']):
        df[col] = df[col].astype('float32')
    return df
```

## Error Handling and Validation

```python
def validate_input(value, valid_range):
    return valid_range <= value <= valid_range

def handle_missing_data(df):
    return df.fillna({
        'ARR_DELAY': df['ARR_DELAY'].mean(),
        'DEP_DELAY': df['DEP_DELAY'].mean()
    })
```

## Testing Framework

```python
def test_metrics():
    """
    Test suite for metric calculations
    """
    test_data = pd.DataFrame({
        'ARR_DELAY': [-20, -10, 0, 10, 20],
        'Status': ['Early', 'On-time', 'On-time', 'On-time', 'Delayed']
    })
    
    assert abs(test_data['ARR_DELAY'].mean()) < 1e-10
```

## Deployment Considerations

### Environment Configuration
```python
import os

def get_config():
    return {
        'data_path': os.getenv('FLIGHT_DATA_PATH', './data/flights.csv'),
        'cache_ttl': int(os.getenv('CACHE_TTL', 3600)),
        'debug_mode': bool(os.getenv('DEBUG_MODE', False))
    }
```

### Performance Monitoring

The system tracks key performance indicators:
\[ \text{Response Time} = T_{\text{end}} - T_{\text{start}} \]
\[ \text{Memory Usage} = M_{\text{used}} / M_{\text{total}} \]

