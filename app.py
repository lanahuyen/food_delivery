import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time
import time as tm

# Must be the first Streamlit command
st.set_page_config(
    page_title="Smart Delivery Timer",
    page_icon="⏱️",
    layout="wide"
)

# Load the saved model and components
model = joblib.load('models/best_delivery_model.pkl')
scaler = joblib.load('models/scaler_delivery.pkl')
feature_names = joblib.load('models/feature_names_delivery.pkl')
metadata = joblib.load('models/model_metadata_delivery.pkl')

# Mid-century modern color palette with increased contrast
COLORS = {
    'primary': '#264653',     # Deep teal
    'secondary': '#e76f51',   # Terracotta
    'accent': '#f4a261',      # Warm peach
    'neutral': '#e9c46a',     # Muted gold
    'background': '#faf9f6',  # Warm white
    'text': '#1a1a1a',        # Darker charcoal for better contrast
    'text_secondary': '#2d4b4d',  # Muted teal for secondary text
    'success': '#2a9d8f'      # Sage green
}

# Industry standards and timing data
TIMING_STANDARDS = {
    'prep_times': {
        'Fast Food': {'min': 5, 'max': 10, 'emoji': '🍔'},
        'Pizza': {'min': 8, 'max': 15, 'emoji': '🍕'},
        'Asian': {'min': 10, 'max': 20, 'emoji': '🍜'},
        'Fine Dining': {'min': 15, 'max': 25, 'emoji': '🍽️'},
        'Casual Dining': {'min': 12, 'max': 20, 'emoji': '🍽️'},
        'Cafe': {'min': 5, 'max': 12, 'emoji': '☕'}
    },
    'weather_factors': {
        'Clear': {'factor': 1.0, 'emoji': '☀️'},
        'Cloudy': {'factor': 1.1, 'emoji': '☁️'},
        'Fog': {'factor': 1.3, 'emoji': '🌫️'},
        'Light Rain': {'factor': 1.2, 'emoji': '🌧️'}
    },
    'traffic_factors': {
        'Low': {'factor': 1.0, 'emoji': '🟢'},
        'Medium': {'factor': 1.2, 'emoji': '🟡'},
        'High': {'factor': 1.4, 'emoji': '🔴'}
    }
}

# Custom CSS with mid-century modern styling and improved contrast
st.markdown("""
    <style>
   /* Overall theme */
    .main {
        background-color: #faf9f6;
        background-image: url('data:image/svg+xml,<svg width="60" height="60" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg"><rect x="0" y="0" width="60" height="60" fill="none" stroke="%23e9e9e9" stroke-width="1"/></svg>');
        color: #2a2a2a;
        font-family: 'Helvetica Neue', sans-serif;
    }

    /* Headers */
    h1, h2, h3 {
        color: #264653 !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
    }

    /* Buttons */
    .stButton>button {
        background-color: #e76f51 !important;
        color: white !important;
        border: none !important;
        border-radius: 2px !important;
        padding: 10px 25px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }

    .stButton>button:hover {
        background-color: #f4a261 !important;
        transform: translateY(-2px);
    }

    /* Metrics */
    .metric-container {
        background-color: white;
        padding: 20px;
        border-radius: 2px;
        border-left: 4px solid #2a9d8f;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }

    /* Selectbox */
    .stSelectbox {
        border-radius: 2px;
    }

    /* Progress bars */
    .stProgress > div > div > div {
        background-color: #2a9d8f;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>Smart Delivery Timer</h1>
        <p style='color: #1a1a1a; font-size: 20px; font-weight: 500;'>Accurate Delivery Time Predictions</p>
    </div>
    """, unsafe_allow_html=True)

# Create main tabs
tab1, tab2 = st.tabs(["Delivery Estimator", "Insights"])

with tab1:
    # Create two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Location & Order")
        
        # Distance slider
        distance = st.slider(
            "Delivery Distance (km)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.1,
            help="Distance is one of the strongest predictors of delivery time"
        )
        
        # Cuisine type selection
        cuisine_options = {k: f"{v['emoji']} {k}" for k, v in TIMING_STANDARDS['prep_times'].items()}
        cuisine_type = st.selectbox(
            "Restaurant Type",
            options=list(cuisine_options.keys()),
            format_func=lambda x: cuisine_options[x]
        )

    with col2:
        st.markdown("### Current Conditions")
        
        # Time of day selection
        time_options = {
            "Early Morning": "7 AM - 10 AM (Quiet hours)",
            "Mid-Morning": "10 AM - 12 PM (Pre-lunch)",
            "Lunch": "12 PM - 2 PM (Lunch rush)",
            "Afternoon": "2 PM - 5 PM (Off-peak)",
            "Dinner": "5 PM - 8 PM (Dinner rush)",
            "Evening": "8 PM - 10 PM (Late dining)"
        }
        time_of_day = st.selectbox(
            "Time of Order",
            options=list(time_options.keys()),
            format_func=lambda x: time_options[x]
        )
        
        # Weather conditions
        weather_options = {k: v['emoji'] + " " + k 
                         for k, v in TIMING_STANDARDS['weather_factors'].items()}
        weather = st.selectbox(
            "Weather",
            options=list(weather_options.keys()),
            format_func=lambda x: weather_options[x]
        )
        
        # Traffic conditions
        traffic_options = {k: v['emoji'] + " " + k + " Traffic" 
                         for k, v in TIMING_STANDARDS['traffic_factors'].items()}
        traffic = st.selectbox(
            "Traffic Conditions",
            options=list(traffic_options.keys()),
            format_func=lambda x: traffic_options[x]
        )

    # Prediction button and results
    if st.button("Calculate Delivery Time", type="primary"):
        try:
            # Create feature vector for ML model
            features = pd.DataFrame(columns=feature_names)
            features.loc[0] = 0
            
            # Set base features
            features['delivery_distance'] = distance
            
            # Set weather conditions
            weather_col = f'Weatherconditions_conditions {weather}'
            if weather_col in features.columns:
                features[weather_col] = 1
            
            # Set traffic density
            traffic_col = f'Road_traffic_density_{traffic} '
            if traffic_col in features.columns:
                features[traffic_col] = 1
            
            # Get base prediction from ML model
            features_scaled = scaler.transform(features)
            base_prediction = model.predict(features_scaled)[0]
            
            # Adjust for cuisine preparation time
            cuisine_prep_time = TIMING_STANDARDS['prep_times'][cuisine_type]['min']
            
            # Time of day factors (now correctly applying increases)
            time_factors = {
                "Early Morning": 0.9,    # 10% faster
                "Mid-Morning": 0.95,     # 5% faster
                "Lunch": 1.3,            # 30% slower
                "Afternoon": 1.0,        # Standard
                "Dinner": 1.3,           # 30% slower
                "Evening": 0.95          # 5% faster
            }
            
            # Calculate final prediction with correct factor application
            time_factor = time_factors[time_of_day]
            weather_factor = TIMING_STANDARDS['weather_factors'][weather]['factor']
            traffic_factor = TIMING_STANDARDS['traffic_factors'][traffic]['factor']
            
            # Apply all factors to get final prediction
            final_prediction = base_prediction
            final_prediction = final_prediction * time_factor      # Apply time of day impact
            final_prediction = final_prediction * weather_factor   # Apply weather impact
            final_prediction = final_prediction * traffic_factor   # Apply traffic impact
            final_prediction = final_prediction + (cuisine_prep_time - 10)  # Add cuisine-specific prep time
            
            # Store in session state
            st.session_state.prediction = final_prediction
            st.session_state.order_details = {
                'cuisine': cuisine_type,
                'base_time': base_prediction,
                'prep_time': cuisine_prep_time,
                'distance': distance,
                'weather': weather,
                'traffic': traffic,
                'time': time_of_day,
                'time_factor': time_factor,
                'weather_factor': weather_factor,
                'traffic_factor': traffic_factor
            }

            # Display Results
            st.markdown("### Estimated Delivery Time")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Total Estimated Time",
                    f"{final_prediction:.0f} minutes",
                    delta=f"Based on {distance:.1f}km distance"
                )
                
                st.markdown("#### Time Breakdown")
                st.write(f"• Preparation: {cuisine_prep_time} minutes")
                st.write(f"• Transit: {(final_prediction - cuisine_prep_time):.0f} minutes")
            
            with col2:
                st.markdown("#### Impact Factors")
                
                # Weather impact (corrected to show increases)
                weather_impact = (weather_factor - 1) * 100
                if weather_impact > 0:
                    st.write(f"• Weather: +{weather_impact:.0f}% ({weather})")
                else:
                    st.write(f"• Weather: Standard time ({weather})")
                
                # Traffic impact (corrected to show increases)
                traffic_impact = (traffic_factor - 1) * 100
                if traffic_impact > 0:
                    st.write(f"• Traffic: +{traffic_impact:.0f}% ({traffic})")
                else:
                    st.write(f"• Traffic: Standard time ({traffic})")
                
                # Time of day impact (corrected to show increases/decreases)
                time_impact = (time_factor - 1) * 100
                if time_impact > 0:
                    st.write(f"• Time of Day: +{time_impact:.0f}% ({time_of_day})")
                elif time_impact < 0:
                    st.write(f"• Time of Day: {time_impact:.0f}% ({time_of_day})")
                else:
                    st.write(f"• Time of Day: Standard time ({time_of_day})")
            
            # Visual timeline
            st.markdown("### Delivery Timeline")
            
            # Create timeline visualization using plotly
            fig = go.Figure()
            
            # Add preparation phase
            fig.add_trace(go.Bar(
                name='Preparation',
                x=[cuisine_prep_time],
                y=['Time'],
                orientation='h',
                marker=dict(color=COLORS['success']),
                text=f"{cuisine_prep_time} min",
                textposition='auto',
            ))
            
            # Add transit phase
            transit_time = final_prediction - cuisine_prep_time
            fig.add_trace(go.Bar(
                name='Transit',
                x=[transit_time],
                y=['Time'],
                orientation='h',
                marker=dict(color=COLORS['secondary']),
                text=f"{transit_time:.0f} min",
                textposition='auto',
            ))
            
            fig.update_layout(
                barmode='stack',
                height=100,
                showlegend=True,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color=COLORS['text']),  # This ensures dark text
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    font=dict(color=COLORS['text'])  # Add this for legend text
                ),
                yaxis=dict(
                    tickfont=dict(color=COLORS['text'])  # Add this for y-axis text
                ),
                xaxis=dict(
                    tickfont=dict(color=COLORS['text'])  # Add this for x-axis text
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error calculating delivery time: {str(e)}")

with tab2:
    st.markdown("### Delivery Insights")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("#### Restaurant Preparation Times")
        
        # Create preparation times comparison
        prep_times_df = pd.DataFrame([
            {
                'Restaurant Type': cuisine,
                'Minimum Time': data['min'],
                'Maximum Time': data['max'],
                'emoji': data['emoji']
            }
            for cuisine, data in TIMING_STANDARDS['prep_times'].items()
        ])

        # Create bar chart for prep times
        fig_prep = go.Figure()
        
        # Add minimum times
        fig_prep.add_trace(go.Bar(
            name='Minimum Time',
            x=prep_times_df['Restaurant Type'],
            y=prep_times_df['Minimum Time'],
            marker_color=COLORS['success'],
            text=prep_times_df['Minimum Time'],
            textposition='auto',
        ))
        
        # Add maximum times
        fig_prep.add_trace(go.Bar(
            name='Maximum Time',
            x=prep_times_df['Restaurant Type'],
            y=prep_times_df['Maximum Time'] - prep_times_df['Minimum Time'],
            marker_color=COLORS['secondary'],
            text=prep_times_df['Maximum Time'],
            textposition='auto',
            base=prep_times_df['Minimum Time']
        ))
        
        fig_prep.update_layout(
            title=dict(
                text='Preparation Time Ranges by Restaurant Type',
                font=dict(color=COLORS['text'])
            ),
            barmode='stack',
            xaxis=dict(
                title='Restaurant Type',
                title_font=dict(color=COLORS['text']),
                tickfont=dict(color=COLORS['text'])
            ),
            yaxis=dict(
                title='Minutes',
                title_font=dict(color=COLORS['text']),
                tickfont=dict(color=COLORS['text'])
            ),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text']),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(color=COLORS['text'])
            )
        )
        
        st.plotly_chart(fig_prep, use_container_width=True)
    
    with insight_col2:
        st.markdown("#### Distance Impact")
        
        # Generate sample predictions for different distances
        distances = np.arange(1, 21, 2)
        times = []
        
        # Calculate delivery times for different distances
        for d in distances:
            features = pd.DataFrame(columns=feature_names)
            features.loc[0] = 0
            features['delivery_distance'] = d
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            times.append(pred)
        
        # Create distance impact visualization
        fig_distance = go.Figure()
        
        fig_distance.add_trace(go.Scatter(
            x=distances,
            y=times,
            mode='lines+markers',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=8, color=COLORS['secondary']),
            name='Delivery Time'
        ))
        
        fig_distance.update_layout(
            title=dict(
                text='How Distance Affects Delivery Time',
                font=dict(color=COLORS['text'])
            ),
            xaxis=dict(
                title='Distance (km)',
                title_font=dict(color=COLORS['text']),
                tickfont=dict(color=COLORS['text'])
            ),
            yaxis=dict(
                title='Base Transit Time (minutes)',
                title_font=dict(color=COLORS['text']),
                tickfont=dict(color=COLORS['text'])
            ),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color=COLORS['text'])
        )
        
        st.plotly_chart(fig_distance, use_container_width=True)

    # Key insights section with corrected impact factors
    st.markdown("### Key Factors")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### Peak Hours Impact
        • Early Morning: -10%
        • Lunch Rush: +30%
        • Afternoon: Standard
        • Dinner Rush: +30%
        • Evening: -5%
        """)
    
    with col2:
        st.markdown("""
        #### Traffic Impact
        • Low: Standard time
        • Medium: +20%
        • High: +40%
        """)
    
    with col3:
        st.markdown("""
        #### Weather Impact
        • Clear: Standard time
        • Cloudy: +10%
        • Fog: +30%
        • Light Rain: +20%
        """)

    # Additional insights about the model
    st.markdown("### About the Predictions")
    st.markdown("""
    This tool combines machine learning predictions with industry standards to provide accurate delivery times:
    
    • Base transit times are calculated using historical delivery data
    • Restaurant preparation times are based on industry standards
    • Weather, traffic, and time of day impacts are applied as multipliers
    • All predictions assume normal operating conditions
    """)

# Footer
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: {COLORS['text']}; padding: 20px;'>
        <p style='font-size: 14px;'>Last updated: {metadata['training_date']}</p>
    </div>
    """, unsafe_allow_html=True)