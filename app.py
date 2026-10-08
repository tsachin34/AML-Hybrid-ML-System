"""AML Detection Dashboard. Run with: streamlit run app.py"""
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
# Add to the imports section
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    accuracy_score  # Add this import
)
from sklearn.metrics import precision_recall_curve, average_precision_score

# Page Configuration
st.set_page_config(
    page_title="AML Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Styling
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        padding: 10px 24px;
        border-radius: 5px;
        border: none;
    }
    </style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent
APP_DATA = BASE_DIR / 'app_data'
MODELS = BASE_DIR / 'models'

@st.cache_data
def load_data():
    try:
        # Load the clustered validation dataset
        df_val = pd.read_csv(APP_DATA / 'aml_df_val_clustered.csv')
        df_original = pd.read_csv(APP_DATA / 'aml_df_original.csv')

        # Verify required columns for validation data
        required_columns = ['Scaled_Amount', 'Scaled_Time', 'Class']
        missing_columns = [col for col in required_columns if col not in df_val.columns]

        if missing_columns:
            raise KeyError(f"Missing required columns: {missing_columns}")

        return df_val, df_original
    except Exception as e:
        st.error(f"Error loading validation data: {str(e)}")
        return None, None

@st.cache_resource
def load_models():
    try:
        # Load all models
        rf_base = joblib.load(MODELS / 'rf_base.joblib')
        rf_enhanced = joblib.load(MODELS / 'rf_enhanced.joblib')
        xgb = joblib.load(MODELS / 'xgb.joblib')
        return rf_base, rf_enhanced, xgb
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

def get_predictions(model_name, X, y, rf_base, rf_enhanced, xgb):
    try:
        if model_name == "Base Random Forest":
            # For base model, remove Cluster if it exists
            X_pred = X.drop(['Cluster'], axis=1) if 'Cluster' in X.columns else X.copy()
            model = rf_base
        elif model_name == "Enhanced Random Forest":
            # For enhanced model, keep all features
            X_pred = X.copy()
            model = rf_enhanced
        else:  # XGBoost
            # For XGBoost, keep all features
            X_pred = X.copy()
            model = xgb

        y_pred = model.predict(X_pred)
        y_pred_proba = model.predict_proba(X_pred)[:, 1]

        return y_pred, y_pred_proba
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.write("Features available:", X.columns.tolist())
        return None, None

def main():
    # Load data and models
    df_val, df_original = load_data()
    rf_base, rf_enhanced, xgb = load_models()

    if df_val is None or df_original is None or rf_base is None:
        return

    # Header
    st.markdown("<h1 style='text-align: center;'>Anti-Money Laundering Detection System</h1>",
                unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 📌 Dashboard Guide")
        with st.expander("How to Use"):
            st.markdown("""
            1. **Model Selection**: Choose between three models:
               - Base Random Forest (Original dataset)
               - Enhanced Random Forest (With clustering)
               - XGBoost (With clustering)

            2. **Filters**:
               - Adjust transaction amount range
               - Filter by time period
               - Select transaction type

            3. **Analysis Tabs**:
               - Model Performance: View ROC curves and metrics
               - Transaction Analysis: Explore patterns
               - Time Series: Analyze temporal trends
               - Real-time Detection: Test transactions
            """)

        st.markdown("### 🎛️ Control Panel")

        selected_model = st.selectbox(
            "Select Model",
            ["Base Random Forest", "Enhanced Random Forest", "XGBoost"],
            help="Choose the model for analysis"
        )

        # Model descriptions
        if selected_model == "Base Random Forest":
            st.info("Basic RF model without clustering")
            working_df = df_original.copy()
        elif selected_model == "Enhanced Random Forest":
            st.info("RF model with clustering enhancement")
            working_df = df_val.copy()
        else:
            st.info("XGBoost model with clustering")
            working_df = df_val.copy()

        # Transaction Type Filter
        transaction_type = st.selectbox(
            "Transaction Type",
            ["All", "Fraud", "Non-Fraud"]
        )

        # Time Range Filter
        time_range = st.slider(
            "Time Period",
            float(working_df['Scaled_Time'].min()),
            float(working_df['Scaled_Time'].max()),
            (float(working_df['Scaled_Time'].min()), float(working_df['Scaled_Time'].max()))
        )

        # Amount Range Filter
        amount_range = st.slider(
            "Transaction Amount",
            float(working_df['Scaled_Amount'].min()),
            float(working_df['Scaled_Amount'].max()),
            (float(working_df['Scaled_Amount'].min()), float(working_df['Scaled_Amount'].max()))
        )

        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.rerun()

    # Apply filters
    filtered_df = working_df[
        (working_df['Scaled_Amount'].between(amount_range[0], amount_range[1])) &
        (working_df['Scaled_Time'].between(time_range[0], time_range[1]))
    ]

    if transaction_type != "All":
        if transaction_type == "Fraud":
            filtered_df = filtered_df[filtered_df['Class'] == 1]
        else:
            filtered_df = filtered_df[filtered_df['Class'] == 0]

    # Key Metrics Display
    st.markdown("### 📊 Key Metrics")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Transactions",
            f"{len(filtered_df):,}",
            delta=f"{len(filtered_df)/len(working_df)*100:.1f}% of total"
        )

    with col2:
        fraud_count = len(filtered_df[filtered_df['Class'] == 1])
        st.metric(
            "Fraud Transactions",
            f"{fraud_count:,}",
            delta=f"{fraud_count/len(filtered_df)*100:.2f}% of filtered"
        )

    with col3:
        avg_amount = filtered_df['Scaled_Amount'].mean()
        st.metric(
            "Average Transaction Amount",
            f"{avg_amount:.2f}",
            delta=f"{avg_amount - working_df['Scaled_Amount'].mean():.2f} vs overall"
        )

    # Updated tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Model Performance",
        "🔍 Transaction Analysis",
        "📈 Time Series",
        "⚡ Real-time Detection"
    ])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            try:
                # ROC Curve (existing code)
                features_for_prediction = filtered_df.drop(['Class'], axis=1)
                y = filtered_df['Class']

                y_pred, y_pred_proba = get_predictions(
                    selected_model,
                    features_for_prediction,
                    y,
                    rf_base,
                    rf_enhanced,
                    xgb
                )

                # ROC Curve
                fpr, tpr, _ = roc_curve(y, y_pred_proba)
                auc_score = roc_auc_score(y, y_pred_proba)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    name=f'ROC (AUC = {auc_score:.3f})'
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    line=dict(dash='dash', color='gray'),
                    name='Random'
                ))
                fig.update_layout(
                    title=f'{selected_model} ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate'
                )
                st.plotly_chart(fig)

                # Precision-Recall Curve
                precision, recall, _ = precision_recall_curve(y, y_pred_proba)
                avg_precision = average_precision_score(y, y_pred_proba)

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=recall, y=precision,
                    name=f'PR (AP = {avg_precision:.3f})'
                ))
                fig.update_layout(
                    title='Precision-Recall Curve',
                    xaxis_title='Recall',
                    yaxis_title='Precision'
                )
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Error in curve generation: {str(e)}")

        with col2:
            try:
                # Confusion Matrix
                cm = confusion_matrix(y, y_pred)
                fig = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=['Non-Fraud', 'Fraud'],
                    y=['Non-Fraud', 'Fraud'],
                    text=cm,
                    texttemplate="%{text}",
                    textfont={"size": 16},
                    colorscale='Blues'
                ))
                fig.update_layout(title='Confusion Matrix')
                st.plotly_chart(fig)

                # Classification Report
                report = classification_report(y, y_pred, output_dict=True)
                df_report = pd.DataFrame(report).transpose()
                st.markdown("### Classification Report")
                st.dataframe(df_report.style.format("{:.3f}"))

                # Additional Metrics
                st.markdown("### Performance Metrics")
                metrics_df = pd.DataFrame({
                    'Metric': ['Accuracy', 'AUC-ROC', 'Avg Precision'],
                    'Value': [
                        accuracy_score(y, y_pred),
                        auc_score,
                        avg_precision
                    ]
                })
                st.table(metrics_df.style.format({'Value': '{:.3f}'}))

            except Exception as e:
                st.error(f"Error in metrics generation: {str(e)}")

    # Feature Importance (if available)
    # Get the correct model based on selection
    if selected_model == "Base Random Forest":
        current_model = rf_base
    elif selected_model == "Enhanced Random Forest":
        current_model = rf_enhanced
    else:
        current_model = xgb

    if hasattr(current_model, 'feature_importances_'):
        st.markdown("### Feature Importance")
        importance = pd.DataFrame({
            'Feature': features_for_prediction.columns,
            'Importance': current_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        fig = px.bar(
            importance.head(10),
            x='Importance',
            y='Feature',
            title=f'Top 10 Important Features - {selected_model}',
            orientation='h'
        )
        st.plotly_chart(fig)

    # Model Comparison (if selected)
    if st.checkbox("Show Model Comparison"):
        st.markdown("### Model Comparison")
        models = {
            "Base Random Forest": rf_base,
            "Enhanced Random Forest": rf_enhanced,
            "XGBoost": xgb
        }

        comparison_metrics = []
        for model_name, model in models.items():
            y_pred, y_pred_proba = get_predictions(
                model_name,
                features_for_prediction,
                y,
                rf_base,
                rf_enhanced,
                xgb
            )
            comparison_metrics.append({
                'Model': model_name,
                'Accuracy': accuracy_score(y, y_pred),
                'AUC-ROC': roc_auc_score(y, y_pred_proba),
                'Avg Precision': average_precision_score(y, y_pred_proba)
            })

        comparison_df = pd.DataFrame(comparison_metrics)
        st.dataframe(comparison_df.style.format({
            'Accuracy': '{:.3f}',
            'AUC-ROC': '{:.3f}',
            'Avg Precision': '{:.3f}'
        }))

    # Transaction Analysis Tab
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(
                filtered_df,
                x='Scaled_Amount',
                color='Class',
                nbins=50,
                title='Transaction Amount Distribution'
            )
            st.plotly_chart(fig)

        with col2:
            fig = px.scatter(
                filtered_df,
                x='Scaled_Amount',
                y='Scaled_Time',
                color='Class',
                title='Transaction Patterns'
            )
            st.plotly_chart(fig)

    # Time Series Analysis Tab
    with tab3:
        time_series = filtered_df.groupby('Scaled_Time')['Class'].mean().rolling(window=50).mean()
        fig = px.line(
            x=time_series.index,
            y=time_series.values,
            title='Fraud Rate Over Time (Moving Average)',
            labels={'x': 'Time', 'y': 'Fraud Rate'}
        )
        st.plotly_chart(fig)

    # Real-time Detection Tab (Placeholder)
    with tab4:
        st.markdown("### ⚡ Real-time Transaction Detection")
        st.info("This feature is currently under development.")
        st.markdown("""
        Future capabilities will include:
        - Real-time transaction analysis
        - Risk scoring
        - Instant fraud detection
        - Alert generation
        """)

# Run the dashboard
main()
