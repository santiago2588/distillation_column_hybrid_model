import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Import our custom models
from first_principles_model import first_principles_distillation_model
from hybrid_model import hybrid_distillation_model

# Page configuration
st.set_page_config(
    page_title="Hybrid Distillation Model",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 2rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .info-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load data and models
@st.cache_data
def load_data():
    """Load the distillation column dataset"""
    try:
        df = pd.read_excel('DistillationColumnDataset.xlsx')
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

@st.cache_resource
def load_models():
    """Load the trained Random Forest model"""
    try:
        rf_model = joblib.load('random_forest_model.joblib')
        with open('rf_model_features.txt', 'r') as f:
            rf_features = [line.strip() for line in f]
        return rf_model, rf_features
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🧪 Hybrid Distillation Column Model</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>📚 Educational Tool for Chemical Engineering</h3>
    This interactive application demonstrates how to build a hybrid model that combines:
    <ul>
        <li><b>First Principles Models</b>: Based on fundamental chemical engineering principles</li>
        <li><b>Machine Learning Models</b>: Trained on historical operational data</li>
        <li><b>Hybrid Approach</b>: Combines both for improved accuracy and interpretability</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("📋 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        [
            "🏠 Overview",
            "📊 Data Exploration", 
            "🔬 First Principles Model",
            "🤖 Machine Learning Model",
            "🔄 Hybrid Model",
            "🎯 Interactive Prediction",
            "📈 Performance Comparison",
            "🎓 Learning Summary"
        ]
    )
    
    # Load data and models
    df = load_data()
    rf_model, rf_features = load_models()
    
    if df is None or rf_model is None:
        st.error("Failed to load data or models. Please check the file paths.")
        return
    
    # Route to different pages
    if page == "🏠 Overview":
        show_overview(df)
    elif page == "📊 Data Exploration":
        show_data_exploration(df)
    elif page == "🔬 First Principles Model":
        show_first_principles_model(df)
    elif page == "🤖 Machine Learning Model":
        show_ml_model(df, rf_model, rf_features)
    elif page == "🔄 Hybrid Model":
        show_hybrid_model(df, rf_model, rf_features)
    elif page == "🎯 Interactive Prediction":
        show_interactive_prediction(df, rf_model, rf_features)
    elif page == "📈 Performance Comparison":
        show_performance_comparison(df, rf_model, rf_features)
    elif page == "🎓 Learning Summary":
        show_learning_summary()

def show_overview(df):
    """Overview page with problem description and methodology"""
    st.markdown('<h2 class="section-header">🎯 Problem Statement</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Objective
        Predict product compositions (**MoleFractionTX** and **MoleFractionHX**) in a distillation column based on operating conditions.
        
        ### Why Hybrid Modeling?
        
        **🔬 First Principles Models:**
        - ✅ Based on fundamental physics and chemistry
        - ✅ Interpretable and explainable
        - ❌ May not capture all real-world complexities
        - ❌ Require detailed knowledge of system parameters
        
        **🤖 Machine Learning Models:**
        - ✅ Learn complex patterns from data
        - ✅ High accuracy on similar operating conditions
        - ❌ "Black box" - difficult to interpret
        - ❌ May not generalize beyond training data
        
        **🔄 Hybrid Models:**
        - ✅ Combine strengths of both approaches
        - ✅ Better accuracy and interpretability
        - ✅ More robust predictions
        """)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>📊 Dataset Overview</h3>
        <ul>
            <li><b>Samples:</b> {}</li>
            <li><b>Features:</b> {}</li>
            <li><b>Target Variables:</b> 2</li>
            <li><b>Time Range:</b> {} hours</li>
        </ul>
        </div>
        """.format(
            df.shape[0],
            df.shape[1] - 3,  # Excluding Time and target variables
            df['Time'].max() / 3600  # Convert seconds to hours
        ), unsafe_allow_html=True)
    
    st.markdown('<h2 class="section-header">🔄 Methodology</h2>', unsafe_allow_html=True)
    
    # Methodology flowchart
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #e3f2fd; border-radius: 0.5rem;">
        <h4>Step 1: Data Analysis</h4>
        <p>Explore the dataset and understand relationships between variables</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #f3e5f5; border-radius: 0.5rem;">
        <h4>Step 2: Model Development</h4>
        <p>Build first principles and ML models independently</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #e8f5e8; border-radius: 0.5rem;">
        <h4>Step 3: Hybrid Integration</h4>
        <p>Combine models with weighted averaging</p>
        </div>
        """, unsafe_allow_html=True)

def show_data_exploration(df):
    """Data exploration page"""
    st.markdown('<h2 class="section-header">📊 Data Exploration</h2>', unsafe_allow_html=True)
    
    # Dataset overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - 3)
    with col3:
        st.metric("Time Span (hours)", f"{df['Time'].max() / 3600:.1f}")
    with col4:
        st.metric("Sampling Interval", "30 seconds")
    
    # Show raw data
    st.subheader("📋 Raw Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    # Target variable distributions
    st.subheader("🎯 Target Variable Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_tx = px.histogram(df, x='MoleFractionTX', nbins=30, 
                             title='Distribution of MoleFractionTX',
                             color_discrete_sequence=['#1f77b4'])
        fig_tx.update_layout(showlegend=False)
        st.plotly_chart(fig_tx, use_container_width=True)
    
    with col2:
        fig_hx = px.histogram(df, x='MoleFractionHX', nbins=30,
                             title='Distribution of MoleFractionHX',
                             color_discrete_sequence=['#ff7f0e'])
        fig_hx.update_layout(showlegend=False)
        st.plotly_chart(fig_hx, use_container_width=True)
    
    # Time series plots
    st.subheader("⏰ Time Series Analysis")
    
    # Convert time to hours for better readability
    df_plot = df.copy()
    df_plot['Time_hours'] = df_plot['Time'] / 3600
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('MoleFractionTX over Time', 'MoleFractionHX over Time'),
        vertical_spacing=0.1
    )
    
    fig.add_trace(
        go.Scatter(x=df_plot['Time_hours'], y=df_plot['MoleFractionTX'],
                  mode='lines', name='MoleFractionTX', line=dict(color='#1f77b4')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=df_plot['Time_hours'], y=df_plot['MoleFractionHX'],
                  mode='lines', name='MoleFractionHX', line=dict(color='#ff7f0e')),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_yaxes(title_text="MoleFractionTX", row=1, col=1)
    fig.update_yaxes(title_text="MoleFractionHX", row=2, col=1)
    fig.update_layout(height=600, showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("🔗 Correlation Analysis")
    
    # Calculate correlation matrix
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    # Focus on correlations with target variables
    target_corr = corr_matrix[['MoleFractionTX', 'MoleFractionHX']].drop(['MoleFractionTX', 'MoleFractionHX'])
    
    fig_corr = px.imshow(target_corr.T, 
                        title='Correlation with Target Variables',
                        color_continuous_scale='RdBu_r',
                        aspect='auto')
    fig_corr.update_layout(height=400)
    st.plotly_chart(fig_corr, use_container_width=True)

def show_first_principles_model(df):
    """First principles model explanation and demonstration"""
    st.markdown('<h2 class="section-header">🔬 First Principles Model</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🧮 Theoretical Foundation</h3>
    First principles models are based on fundamental chemical engineering principles:
    <ul>
        <li><b>Mass Balance:</b> Conservation of mass for each component</li>
        <li><b>Energy Balance:</b> Conservation of energy across the column</li>
        <li><b>Vapor-Liquid Equilibrium:</b> Thermodynamic relationships</li>
        <li><b>Heat and Mass Transfer:</b> Rate-based phenomena</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Simplified model explanation
    st.subheader("📝 Simplified Model Implementation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        For this demonstration, we use a simplified first principles model that includes:
        
        **1. Antoine Equation for Vapor Pressure:**
        ```
        log₁₀(P_vap) = A - (B / (T + C))
        ```
        
        **2. Raoult's Law for Ideal Mixtures:**
        ```
        y_i = (x_i × P_vap_i) / P_total
        ```
        
        **3. Simplified Mass Balance:**
        ```
        Enrichment ∝ f(Reflux Ratio, Relative Volatility)
        ```
        
        **Note:** This is a highly simplified representation. Real industrial models would use:
        - MESH equations (Mass, Equilibrium, Summation, Heat)
        - Activity coefficient models (NRTL, UNIQUAC)
        - Stage-by-stage calculations
        """)
    
    with col2:
        st.markdown("""
        <div class="warning-box">
        <h4>⚠️ Model Limitations</h4>
        <p>This simplified model is for educational purposes. Real applications require:</p>
        <ul>
            <li>Rigorous thermodynamic models</li>
            <li>Detailed equipment specifications</li>
            <li>Non-ideal behavior considerations</li>
            <li>Dynamic effects</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive demonstration
    st.subheader("🎮 Interactive Demonstration")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Adjust Parameters:**")
        feed_mf = st.slider("Feed Mole Fraction", 0.1, 0.9, 0.5, 0.01)
        reflux_ratio = st.slider("Reflux Ratio", 0.5, 5.0, 2.0, 0.1)
        feed_temp = st.slider("Feed Temperature (°C)", 50, 150, 100, 1)
        cond_pressure = st.slider("Condenser Pressure (kPa)", 80, 120, 100, 1)
        reboiler_pressure = st.slider("Reboiler Pressure (kPa)", 100, 130, 110, 1)
    
    with col2:
        if st.button("🔬 Calculate First Principles Prediction", type="primary"):
            try:
                fp_tx, fp_hx = first_principles_distillation_model(
                    feed_mf, reflux_ratio, 20, cond_pressure, reboiler_pressure, feed_temp
                )
                
                st.success("✅ Calculation Complete!")
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("MoleFractionTX", f"{fp_tx:.6f}")
                with col_b:
                    st.metric("MoleFractionHX", f"{fp_hx:.6f}")
                
                # Visualization
                fig = go.Figure(data=[
                    go.Bar(name='Predicted Composition', 
                          x=['MoleFractionTX', 'MoleFractionHX'], 
                          y=[fp_tx, fp_hx],
                          marker_color=['#1f77b4', '#ff7f0e'])
                ])
                fig.update_layout(title='First Principles Prediction', yaxis_title='Mole Fraction')
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error in calculation: {str(e)}")

def show_ml_model(df, rf_model, rf_features):
    """Machine learning model explanation and performance"""
    st.markdown('<h2 class="section-header">🤖 Machine Learning Model</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🌳 Random Forest Regressor</h3>
    We use a Random Forest model because it:
    <ul>
        <li><b>Handles non-linear relationships</b> naturally</li>
        <li><b>Provides feature importance</b> for interpretability</li>
        <li><b>Robust to outliers</b> and missing data</li>
        <li><b>Works well with tabular data</b> like process data</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Model performance
    st.subheader("📊 Model Performance")
    
    # Prepare data for evaluation
    X = df.drop(['Time', 'MoleFractionTX', 'MoleFractionHX'], axis=1)
    y_true = df[['MoleFractionTX', 'MoleFractionHX']]
    
    # Make predictions
    rf_predictions = rf_model.predict(X)
    
    # Calculate metrics
    mse = mean_squared_error(y_true, rf_predictions)
    r2 = r2_score(y_true, rf_predictions)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R² Score", f"{r2:.4f}")
    with col2:
        st.metric("MSE", f"{mse:.2e}")
    with col3:
        st.metric("RMSE", f"{np.sqrt(mse):.6f}")
    with col4:
        st.metric("Training Samples", len(df))
    
    # Prediction vs Actual plots
    st.subheader("🎯 Predictions vs Actual Values")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_tx = px.scatter(
            x=y_true['MoleFractionTX'], 
            y=rf_predictions[:, 0],
            title='MoleFractionTX: Predicted vs Actual',
            labels={'x': 'Actual', 'y': 'Predicted'}
        )
        # Add perfect prediction line
        min_val = min(y_true['MoleFractionTX'].min(), rf_predictions[:, 0].min())
        max_val = max(y_true['MoleFractionTX'].max(), rf_predictions[:, 0].max())
        fig_tx.add_shape(
            type="line", line=dict(dash="dash", color="red"),
            x0=min_val, y0=min_val, x1=max_val, y1=max_val
        )
        st.plotly_chart(fig_tx, use_container_width=True)
    
    with col2:
        fig_hx = px.scatter(
            x=y_true['MoleFractionHX'], 
            y=rf_predictions[:, 1],
            title='MoleFractionHX: Predicted vs Actual',
            labels={'x': 'Actual', 'y': 'Predicted'}
        )
        # Add perfect prediction line
        min_val = min(y_true['MoleFractionHX'].min(), rf_predictions[:, 1].min())
        max_val = max(y_true['MoleFractionHX'].max(), rf_predictions[:, 1].max())
        fig_hx.add_shape(
            type="line", line=dict(dash="dash", color="red"),
            x0=min_val, y0=min_val, x1=max_val, y1=max_val
        )
        st.plotly_chart(fig_hx, use_container_width=True)
    
    # Feature importance
    st.subheader("🔍 Feature Importance Analysis")
    
    feature_importance = pd.DataFrame({
        'Feature': rf_features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig_importance = px.bar(
        feature_importance, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Random Forest Feature Importance',
        color='Importance',
        color_continuous_scale='viridis'
    )
    fig_importance.update_layout(height=600)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Top features explanation
    st.markdown("**🏆 Top 5 Most Important Features:**")
    top_features = feature_importance.tail(5)
    for i, (_, row) in enumerate(top_features.iterrows(), 1):
        st.write(f"{i}. **{row['Feature']}**: {row['Importance']:.4f}")

def show_hybrid_model(df, rf_model, rf_features):
    """Hybrid model explanation and methodology"""
    st.markdown('<h2 class="section-header">🔄 Hybrid Model</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Hybrid Approach</h3>
    The hybrid model combines first principles and machine learning predictions using weighted averaging:
    <br><br>
    <b>Hybrid Prediction = α × First_Principles + (1-α) × Machine_Learning</b>
    <br><br>
    Where α is the weighting parameter (0 ≤ α ≤ 1)
    </div>
    """, unsafe_allow_html=True)
    
    # Alpha parameter explanation
    st.subheader("⚖️ Weighting Parameter (α)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #e3f2fd; border-radius: 0.5rem;">
        <h4>α = 0</h4>
        <p><b>Pure ML Model</b></p>
        <p>Best for conditions similar to training data</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #f3e5f5; border-radius: 0.5rem;">
        <h4>α = 0.5</h4>
        <p><b>Balanced Hybrid</b></p>
        <p>Equal weight to both approaches</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background-color: #e8f5e8; border-radius: 0.5rem;">
        <h4>α = 1</h4>
        <p><b>Pure First Principles</b></p>
        <p>Best for extrapolation beyond training data</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Interactive alpha comparison
    st.subheader("🎮 Interactive Alpha Comparison")
    
    # Select a sample from the dataset
    sample_idx = st.selectbox("Select a data sample:", range(0, min(100, len(df)), 10))
    sample_data = df.iloc[sample_idx]
    
    # Prepare operating conditions
    operating_conditions = {col: sample_data[col] for col in rf_features}
    
    # Alpha slider
    alpha = st.slider("Alpha (α) - Weighting Parameter", 0.0, 1.0, 0.5, 0.1)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Selected Sample Conditions:**")
        for feature in rf_features[:7]:  # Show first 7 features
            st.write(f"**{feature}:** {operating_conditions[feature]:.3f}")
        if len(rf_features) > 7:
            st.write(f"... and {len(rf_features) - 7} more features")
    
    with col2:
        try:
            # Get individual predictions
            rf_input = pd.DataFrame([operating_conditions])
            ml_pred = rf_model.predict(rf_input)[0]
            
            fp_pred_tx, fp_pred_hx = first_principles_distillation_model(
                operating_conditions['Feed Mole Fraction'],
                operating_conditions['Reflux Ratio'],
                20,
                operating_conditions['Condenser Pressure'],
                operating_conditions['Bottom Tower Pressure'],
                operating_conditions['Feed Tray Temperature']
            )
            
            # Calculate hybrid prediction
            hybrid_tx = alpha * fp_pred_tx + (1 - alpha) * ml_pred[0]
            hybrid_hx = alpha * fp_pred_hx + (1 - alpha) * ml_pred[1]
            
            # Actual values
            actual_tx = sample_data['MoleFractionTX']
            actual_hx = sample_data['MoleFractionHX']
            
            # Create comparison chart
            models = ['First Principles', 'Machine Learning', 'Hybrid', 'Actual']
            tx_values = [fp_pred_tx, ml_pred[0], hybrid_tx, actual_tx]
            hx_values = [fp_pred_hx, ml_pred[1], hybrid_hx, actual_hx]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('MoleFractionTX Comparison', 'MoleFractionHX Comparison')
            )
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            
            fig.add_trace(
                go.Bar(x=models, y=tx_values, name='TX', marker_color=colors),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=models, y=hx_values, name='HX', marker_color=colors),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Error analysis
            st.markdown("**📊 Error Analysis:**")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                fp_error = abs(fp_pred_tx - actual_tx) + abs(fp_pred_hx - actual_hx)
                st.metric("First Principles Error", f"{fp_error:.6f}")
            
            with col_b:
                ml_error = abs(ml_pred[0] - actual_tx) + abs(ml_pred[1] - actual_hx)
                st.metric("ML Error", f"{ml_error:.6f}")
            
            with col_c:
                hybrid_error = abs(hybrid_tx - actual_tx) + abs(hybrid_hx - actual_hx)
                st.metric("Hybrid Error", f"{hybrid_error:.6f}")
                
        except Exception as e:
            st.error(f"Error in calculation: {str(e)}")

def show_interactive_prediction(df, rf_model, rf_features):
    """Interactive prediction interface"""
    st.markdown('<h2 class="section-header">🎯 Interactive Prediction</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎮 Try Your Own Predictions!</h3>
    Adjust the operating parameters below to see how different models predict the product compositions.
    This is perfect for exploring "what-if" scenarios in your distillation column operation.
    </div>
    """, unsafe_allow_html=True)
    
    # Get feature ranges for sliders
    feature_ranges = {}
    for col in rf_features:
        feature_ranges[col] = (df[col].min(), df[col].max(), df[col].mean())
    
    # Create input interface
    st.subheader("🎛️ Operating Parameters")
    
    # Organize parameters into logical groups
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🌡️ Temperature & Pressure**")
        feed_temp = st.slider(
            "Feed Tray Temperature",
            float(feature_ranges['Feed Tray Temperature'][0]),
            float(feature_ranges['Feed Tray Temperature'][1]),
            float(feature_ranges['Feed Tray Temperature'][2]),
            0.1
        )
        
        cond_pressure = st.slider(
            "Condenser Pressure",
            float(feature_ranges['Condenser Pressure'][0]),
            float(feature_ranges['Condenser Pressure'][1]),
            float(feature_ranges['Condenser Pressure'][2]),
            0.1
        )
        
        main_pressure = st.slider(
            "Main Tower Pressure",
            float(feature_ranges['Main Tower Pressure'][0]),
            float(feature_ranges['Main Tower Pressure'][1]),
            float(feature_ranges['Main Tower Pressure'][2]),
            0.1
        )
        
        bottom_pressure = st.slider(
            "Bottom Tower Pressure",
            float(feature_ranges['Bottom Tower Pressure'][0]),
            float(feature_ranges['Bottom Tower Pressure'][1]),
            float(feature_ranges['Bottom Tower Pressure'][2]),
            0.1
        )
        
        top_pressure = st.slider(
            "Top Tower Pressure",
            float(feature_ranges['Top Tower Pressure'][0]),
            float(feature_ranges['Top Tower Pressure'][1]),
            float(feature_ranges['Top Tower Pressure'][2]),
            0.1
        )
    
    with col2:
        st.markdown("**💧 Flow Rates & Levels**")
        feed_flow = st.slider(
            "Mass Flow Rate in Feed Flow",
            float(feature_ranges['Mass Flow Rate in Feed Flow'][0]),
            float(feature_ranges['Mass Flow Rate in Feed Flow'][1]),
            float(feature_ranges['Mass Flow Rate in Feed Flow'][2]),
            1.0
        )
        
        top_outlet_flow = st.slider(
            "Mass Flow Rate in Top Outlet Stream",
            float(feature_ranges['Mass Flow Rate in Top Outlet Stream'][0]),
            float(feature_ranges['Mass Flow Rate in Top Outlet Stream'][1]),
            float(feature_ranges['Mass Flow Rate in Top Outlet Stream'][2]),
            1.0
        )
        
        net_mass_flow = st.slider(
            "Net Mass Flow in main tower",
            float(feature_ranges['Net Mass Flow in main tower'][0]),
            float(feature_ranges['Net Mass Flow in main tower'][1]),
            float(feature_ranges['Net Mass Flow in main tower'][2]),
            1.0
        )
        
        liquid_condenser = st.slider(
            "Liquid Percentage in Condenser",
            float(feature_ranges['Liquid Percentage in Condenser'][0]),
            float(feature_ranges['Liquid Percentage in Condenser'][1]),
            float(feature_ranges['Liquid Percentage in Condenser'][2]),
            0.1
        )
        
        liquid_reboiler = st.slider(
            "Liquid Percentage in Reboiler",
            float(feature_ranges['Liquid Percentage in Reboiler'][0]),
            float(feature_ranges['Liquid Percentage in Reboiler'][1]),
            float(feature_ranges['Liquid Percentage in Reboiler'][2]),
            0.1
        )
    
    with col3:
        st.markdown("**🧪 Compositions & Control**")
        feed_mole_fraction = st.slider(
            "Feed Mole Fraction",
            float(feature_ranges['Feed Mole Fraction'][0]),
            float(feature_ranges['Feed Mole Fraction'][1]),
            float(feature_ranges['Feed Mole Fraction'][2]),
            0.001
        )
        
        hx_reboiler = st.slider(
            "Mole Fraction HX at reboiler",
            float(feature_ranges['Mole Fraction HX at reboiler'][0]),
            float(feature_ranges['Mole Fraction HX at reboiler'][1]),
            float(feature_ranges['Mole Fraction HX at reboiler'][2]),
            0.001
        )
        
        hx_top_outlet = st.slider(
            "HX Mole Fraction in Top Outler Stream",
            float(feature_ranges['HX Mole Fraction in Top Outler Stream'][0]),
            float(feature_ranges['HX Mole Fraction in Top Outler Stream'][1]),
            float(feature_ranges['HX Mole Fraction in Top Outler Stream'][2]),
            0.001
        )
        
        reflux_ratio = st.slider(
            "Reflux Ratio",
            float(feature_ranges['Reflux Ratio'][0]),
            float(feature_ranges['Reflux Ratio'][1]),
            float(feature_ranges['Reflux Ratio'][2]),
            0.1
        )
        
        # Alpha parameter for hybrid model
        st.markdown("**⚖️ Hybrid Model Weight**")
        alpha = st.slider(
            "Alpha (α) - First Principles Weight",
            0.0, 1.0, 0.5, 0.1,
            help="0 = Pure ML, 1 = Pure First Principles"
        )
    
    # Prediction button
    if st.button("🔮 Make Prediction", type="primary", use_container_width=True):
        # Prepare operating conditions
        operating_conditions = {
            'Liquid Percentage in Condenser': liquid_condenser,
            'Condenser Pressure': cond_pressure,
            'Liquid Percentage in Reboiler': liquid_reboiler,
            'Mass Flow Rate in Feed Flow': feed_flow,
            'Mass Flow Rate in Top Outlet Stream': top_outlet_flow,
            'Net Mass Flow in main tower': net_mass_flow,
            'Mole Fraction HX at reboiler': hx_reboiler,
            'HX Mole Fraction in Top Outler Stream': hx_top_outlet,
            'Feed Mole Fraction': feed_mole_fraction,
            'Feed Tray Temperature': feed_temp,
            'Main Tower Pressure': main_pressure,
            'Bottom Tower Pressure': bottom_pressure,
            'Top Tower Pressure': top_pressure,
            'Reflux Ratio': reflux_ratio
        }
        
        try:
            # Get predictions from all models
            rf_input = pd.DataFrame([operating_conditions])
            ml_pred = rf_model.predict(rf_input)[0]
            
            fp_pred_tx, fp_pred_hx = first_principles_distillation_model(
                feed_mole_fraction, reflux_ratio, 20, cond_pressure, bottom_pressure, feed_temp
            )
            
            hybrid_tx, hybrid_hx = hybrid_distillation_model(operating_conditions, alpha)
            
            # Display results
            st.markdown('<h3 class="section-header">🎯 Prediction Results</h3>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                <div class="metric-card">
                <h4>🔬 First Principles</h4>
                <p><b>MoleFractionTX:</b> {:.6f}</p>
                <p><b>MoleFractionHX:</b> {:.6f}</p>
                </div>
                """.format(fp_pred_tx, fp_pred_hx), unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                <h4>🤖 Machine Learning</h4>
                <p><b>MoleFractionTX:</b> {:.6f}</p>
                <p><b>MoleFractionHX:</b> {:.6f}</p>
                </div>
                """.format(ml_pred[0], ml_pred[1]), unsafe_allow_html=True)
            
            with col3:
                st.markdown("""
                <div class="metric-card">
                <h4>🔄 Hybrid (α={:.1f})</h4>
                <p><b>MoleFractionTX:</b> {:.6f}</p>
                <p><b>MoleFractionHX:</b> {:.6f}</p>
                </div>
                """.format(alpha, hybrid_tx, hybrid_hx), unsafe_allow_html=True)
            
            # Visualization
            models = ['First Principles', 'Machine Learning', 'Hybrid']
            tx_values = [fp_pred_tx, ml_pred[0], hybrid_tx]
            hx_values = [fp_pred_hx, ml_pred[1], hybrid_hx]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('MoleFractionTX Predictions', 'MoleFractionHX Predictions')
            )
            
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
            
            fig.add_trace(
                go.Bar(x=models, y=tx_values, name='TX', marker_color=colors),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=models, y=hx_values, name='HX', marker_color=colors),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Model differences
            st.subheader("📊 Model Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                tx_diff_fp_ml = abs(fp_pred_tx - ml_pred[0])
                st.metric("TX Difference (FP vs ML)", f"{tx_diff_fp_ml:.6f}")
            
            with col2:
                hx_diff_fp_ml = abs(fp_pred_hx - ml_pred[1])
                st.metric("HX Difference (FP vs ML)", f"{hx_diff_fp_ml:.6f}")
                
        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")

def show_performance_comparison(df, rf_model, rf_features):
    """Performance comparison across all models"""
    st.markdown('<h2 class="section-header">📈 Performance Comparison</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🏆 Model Evaluation</h3>
    Compare the performance of all three modeling approaches across the entire dataset.
    This helps understand when each approach works best.
    </div>
    """, unsafe_allow_html=True)
    
    # Prepare data
    X = df.drop(['Time', 'MoleFractionTX', 'MoleFractionHX'], axis=1)
    y_true = df[['MoleFractionTX', 'MoleFractionHX']].values
    
    # Get ML predictions
    ml_predictions = rf_model.predict(X)
    
    # Get first principles predictions (simplified for all samples)
    fp_predictions = []
    hybrid_predictions_05 = []
    hybrid_predictions_02 = []
    hybrid_predictions_08 = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (_, row) in enumerate(df.iterrows()):
        if i % 100 == 0:
            progress_bar.progress(i / len(df))
            status_text.text(f'Calculating predictions... {i}/{len(df)}')
        
        try:
            fp_tx, fp_hx = first_principles_distillation_model(
                row['Feed Mole Fraction'],
                row['Reflux Ratio'],
                20,
                row['Condenser Pressure'],
                row['Bottom Tower Pressure'],
                row['Feed Tray Temperature']
            )
            fp_predictions.append([fp_tx, fp_hx])
            
            # Hybrid predictions with different alpha values
            operating_conditions = {col: row[col] for col in rf_features}
            
            h_tx_05, h_hx_05 = hybrid_distillation_model(operating_conditions, 0.5)
            hybrid_predictions_05.append([h_tx_05, h_hx_05])
            
            h_tx_02, h_hx_02 = hybrid_distillation_model(operating_conditions, 0.2)
            hybrid_predictions_02.append([h_tx_02, h_hx_02])
            
            h_tx_08, h_hx_08 = hybrid_distillation_model(operating_conditions, 0.8)
            hybrid_predictions_08.append([h_tx_08, h_hx_08])
            
        except:
            # Fallback for any calculation errors
            fp_predictions.append([0.5, 0.5])
            hybrid_predictions_05.append([0.5, 0.5])
            hybrid_predictions_02.append([0.5, 0.5])
            hybrid_predictions_08.append([0.5, 0.5])
    
    progress_bar.progress(1.0)
    status_text.text('Calculations complete!')
    
    fp_predictions = np.array(fp_predictions)
    hybrid_predictions_05 = np.array(hybrid_predictions_05)
    hybrid_predictions_02 = np.array(hybrid_predictions_02)
    hybrid_predictions_08 = np.array(hybrid_predictions_08)
    
    # Calculate metrics
    models = {
        'Machine Learning': ml_predictions,
        'First Principles': fp_predictions,
        'Hybrid (α=0.2)': hybrid_predictions_02,
        'Hybrid (α=0.5)': hybrid_predictions_05,
        'Hybrid (α=0.8)': hybrid_predictions_08
    }
    
    metrics_data = []
    for model_name, predictions in models.items():
        mse = mean_squared_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)
        rmse = np.sqrt(mse)
        
        metrics_data.append({
            'Model': model_name,
            'MSE': mse,
            'RMSE': rmse,
            'R²': r2
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Display metrics table
    st.subheader("📊 Performance Metrics")
    st.dataframe(metrics_df.round(6), use_container_width=True)
    
    # Visualize metrics
    col1, col2 = st.columns(2)
    
    with col1:
        fig_r2 = px.bar(metrics_df, x='Model', y='R²', 
                       title='R² Score Comparison',
                       color='R²', color_continuous_scale='viridis')
        fig_r2.update_xaxes(tickangle=45)
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        fig_rmse = px.bar(metrics_df, x='Model', y='RMSE',
                         title='RMSE Comparison',
                         color='RMSE', color_continuous_scale='viridis_r')
        fig_rmse.update_xaxes(tickangle=45)
        st.plotly_chart(fig_rmse, use_container_width=True)
    
    # Best model identification
    best_r2_model = metrics_df.loc[metrics_df['R²'].idxmax(), 'Model']
    best_rmse_model = metrics_df.loc[metrics_df['RMSE'].idxmin(), 'Model']
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 **Best R² Score:** {best_r2_model}")
    with col2:
        st.success(f"🎯 **Lowest RMSE:** {best_rmse_model}")
    
    # Prediction scatter plots
    st.subheader("🎯 Prediction Accuracy Visualization")
    
    model_to_plot = st.selectbox("Select model to visualize:", list(models.keys()))
    selected_predictions = models[model_to_plot]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_tx = px.scatter(
            x=y_true[:, 0], y=selected_predictions[:, 0],
            title=f'{model_to_plot}: MoleFractionTX',
            labels={'x': 'Actual', 'y': 'Predicted'},
            opacity=0.6
        )
        # Add perfect prediction line
        min_val = min(y_true[:, 0].min(), selected_predictions[:, 0].min())
        max_val = max(y_true[:, 0].max(), selected_predictions[:, 0].max())
        fig_tx.add_shape(
            type="line", line=dict(dash="dash", color="red"),
            x0=min_val, y0=min_val, x1=max_val, y1=max_val
        )
        st.plotly_chart(fig_tx, use_container_width=True)
    
    with col2:
        fig_hx = px.scatter(
            x=y_true[:, 1], y=selected_predictions[:, 1],
            title=f'{model_to_plot}: MoleFractionHX',
            labels={'x': 'Actual', 'y': 'Predicted'},
            opacity=0.6
        )
        # Add perfect prediction line
        min_val = min(y_true[:, 1].min(), selected_predictions[:, 1].min())
        max_val = max(y_true[:, 1].max(), selected_predictions[:, 1].max())
        fig_hx.add_shape(
            type="line", line=dict(dash="dash", color="red"),
            x0=min_val, y0=min_val, x1=max_val, y1=max_val
        )
        st.plotly_chart(fig_hx, use_container_width=True)

def show_learning_summary():
    """Learning summary and key takeaways"""
    st.markdown('<h2 class="section-header">🎓 Learning Summary</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🎯 Key Learning Objectives</h3>
    This application demonstrates the power of hybrid modeling in chemical engineering process control and optimization.
    </div>
    """, unsafe_allow_html=True)
    
    # Key concepts
    st.subheader("📚 Key Concepts Learned")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔬 First Principles Modeling:**
        - Based on fundamental physics and chemistry
        - Uses mass and energy balances
        - Incorporates thermodynamic relationships
        - Provides physical insight and interpretability
        - Can extrapolate beyond training data
        - Requires detailed system knowledge
        
        **🤖 Machine Learning Modeling:**
        - Learns patterns from historical data
        - Handles complex non-linear relationships
        - High accuracy on similar conditions
        - Requires minimal domain knowledge
        - Limited extrapolation capability
        - "Black box" nature
        """)
    
    with col2:
        st.markdown("""
        **🔄 Hybrid Modeling:**
        - Combines strengths of both approaches
        - Balances accuracy and interpretability
        - More robust predictions
        - Flexible weighting system
        - Better uncertainty handling
        - Industry best practice
        
        **⚖️ Alpha Parameter:**
        - Controls model weighting
        - α = 0: Pure machine learning
        - α = 1: Pure first principles
        - α = 0.5: Balanced approach
        - Can be optimized for specific conditions
        """)
    
    # When to use each approach
    st.subheader("🎯 When to Use Each Approach")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="padding: 1rem; background-color: #e3f2fd; border-radius: 0.5rem;">
        <h4>🔬 Use First Principles When:</h4>
        <ul>
            <li>Limited historical data</li>
            <li>New operating conditions</li>
            <li>Need physical understanding</li>
            <li>Regulatory requirements</li>
            <li>Safety-critical applications</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="padding: 1rem; background-color: #f3e5f5; border-radius: 0.5rem;">
        <h4>🤖 Use Machine Learning When:</h4>
        <ul>
            <li>Abundant historical data</li>
            <li>Complex system behavior</li>
            <li>Similar operating conditions</li>
            <li>High accuracy requirements</li>
            <li>Real-time applications</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="padding: 1rem; background-color: #e8f5e8; border-radius: 0.5rem;">
        <h4>🔄 Use Hybrid When:</h4>
        <ul>
            <li>Best of both worlds needed</li>
            <li>Varying operating conditions</li>
            <li>Uncertainty quantification</li>
            <li>Model validation required</li>
            <li>Production environments</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Implementation considerations
    st.subheader("⚙️ Implementation Considerations")
    
    st.markdown("""
    **🔧 Technical Considerations:**
    - **Data Quality**: Ensure clean, representative training data
    - **Feature Engineering**: Select relevant process variables
    - **Model Validation**: Use proper cross-validation techniques
    - **Uncertainty Quantification**: Implement confidence intervals
    - **Real-time Performance**: Consider computational requirements
    - **Model Maintenance**: Plan for model updates and retraining
    
    **🏭 Industrial Applications:**
    - **Process Optimization**: Maximize efficiency and yield
    - **Quality Control**: Maintain product specifications
    - **Predictive Maintenance**: Anticipate equipment issues
    - **Safety Monitoring**: Detect abnormal conditions
    - **Energy Management**: Optimize utility consumption
    - **Environmental Compliance**: Monitor emissions and waste
    """)
    
    # Future directions
    st.subheader("🚀 Future Directions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔬 Advanced Modeling Techniques:**
        - Physics-informed neural networks (PINNs)
        - Gaussian process regression
        - Ensemble methods
        - Deep learning architectures
        - Reinforcement learning for control
        """)
    
    with col2:
        st.markdown("""
        **🏭 Industry 4.0 Integration:**
        - Digital twins
        - Edge computing
        - IoT sensor integration
        - Cloud-based analytics
        - Automated model deployment
        """)
    
    # Quiz section
    st.subheader("🧠 Knowledge Check")
    
    with st.expander("Click to test your understanding!"):
        q1 = st.radio(
            "What is the main advantage of hybrid modeling?",
            ["Higher computational speed", "Combines strengths of different approaches", "Requires less data", "Always more accurate"]
        )
        
        if q1 == "Combines strengths of different approaches":
            st.success("✅ Correct! Hybrid models leverage the interpretability of first principles and the accuracy of ML.")
        elif q1:
            st.error("❌ Try again! Think about what makes hybrid models unique.")
        
        q2 = st.radio(
            "When should you use α = 1 in the hybrid model?",
            ["When you have lots of data", "When extrapolating beyond training data", "When you need fast predictions", "When accuracy is not important"]
        )
        
        if q2 == "When extrapolating beyond training data":
            st.success("✅ Correct! First principles models are better for extrapolation beyond known conditions.")
        elif q2:
            st.error("❌ Try again! Consider when first principles models are most valuable.")
    
    # Resources
    st.subheader("📖 Additional Resources")
    
    st.markdown("""
    **📚 Recommended Reading:**
    - "Process Modeling and Simulation" by Aspen Technology
    - "Machine Learning for Chemical Engineering" by various authors
    - "Hybrid Modeling in Process Industries" - AIChE Journal
    - "Digital Twins in Chemical Engineering" - Computers & Chemical Engineering
    
    **🔗 Useful Links:**
    - NIST Chemistry WebBook (thermodynamic data)
    - Scikit-learn documentation (machine learning)
    - Process Systems Engineering community
    - AIChE (American Institute of Chemical Engineers)
    """)

if __name__ == "__main__":
    main()

