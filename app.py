
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Obesity Level Classification",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# LOAD MODEL FILES
# ============================================================

@st.cache_resource
def load_model_files():
    base_dir = Path(__file__).resolve().parent

    model = joblib.load(base_dir / "best_obesity_model.pkl")
    scaler = joblib.load(base_dir / "scaler.pkl")
    label_encoder = joblib.load(base_dir / "label_encoder.pkl")
    feature_columns = joblib.load(base_dir / "feature_columns.pkl")

    return model, scaler, label_encoder, feature_columns


try:
    model, scaler, label_encoder, feature_columns = load_model_files()
except Exception as e:
    st.error("Unable to load the trained model or preprocessing files.")
    st.code(str(e))
    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Obesity Classification")

page = st.sidebar.radio(
    "Presentation Navigation",
    [
        "Project Overview",
        "Data Preparation",
        "Model Comparison",
        "Live Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    f"Loaded model: {type(model).__name__}"
)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":

    st.title("📊 Obesity Level Classification")

    st.write(
        """
        This project applies machine learning techniques to classify
        individuals into different obesity levels based on physical,
        dietary, and lifestyle attributes.
        """
    )

    st.info(
        "The system compares three machine learning models and deploys "
        "the best-performing model in this Streamlit prototype."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", "1,612")
    col2.metric("Input Features", "16")
    col3.metric("Encoded Features", "22")
    col4.metric("Target Classes", "7")

    st.subheader("Target Classes")

    class_df = pd.DataFrame(
        {
            "Obesity Level": [
                "Insufficient_Weight",
                "Normal_Weight",
                "Overweight_Level_I",
                "Overweight_Level_II",
                "Obesity_Type_I",
                "Obesity_Type_II",
                "Obesity_Type_III"
            ],
            "Records": [
                160,
                160,
                160,
                160,
                351,
                297,
                324
            ]
        }
    )

    st.dataframe(
        class_df,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        class_df.set_index("Obesity Level")["Records"]
    )

    st.subheader("Models Used")

    model_df = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "K-Nearest Neighbors"
            ],
            "Role": [
                "Baseline",
                "Tuned Model",
                "Tuned Model"
            ],
            "Tuning": [
                "Fixed parameters",
                "GridSearchCV",
                "GridSearchCV"
            ]
        }
    )

    st.dataframe(
        model_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DATA PREPARATION
# ============================================================

elif page == "Data Preparation":

    st.title("🧹 Data Preparation")

    st.write(
        """
        The dataset was cleaned and transformed before model training.
        The same preprocessing objects are reused in the Streamlit app
        to keep prediction consistent with the training process.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Training Samples", "1,289")
    col2.metric("Testing Samples", "323")
    col3.metric("Train / Test Split", "80% / 20%")

    st.subheader("Main Preprocessing Steps")

    preprocessing_df = pd.DataFrame(
        {
            "Step": [
                "1",
                "2",
                "3",
                "4",
                "5",
                "6"
            ],
            "Process": [
                "Data cleaning",
                "Target encoding",
                "Categorical encoding",
                "Train-test split",
                "Feature standardization",
                "Save preprocessing objects"
            ],
            "Method": [
                "Check missing values, duplicates and outliers",
                "LabelEncoder",
                "One-hot encoding with drop_first=True",
                "80/20 stratified split, random_state=42",
                "StandardScaler fitted on training data",
                "scaler.pkl, label_encoder.pkl, feature_columns.pkl"
            ]
        }
    )

    st.dataframe(
        preprocessing_df,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "Important: the scaler is fitted only on the training set, "
        "then used to transform both the test set and new Streamlit input."
    )

    with st.expander("Encoded feature columns used by the model"):
        st.write(feature_columns)


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.title("🤖 Model Comparison")

    st.write(
        """
        Logistic Regression is used as the baseline model.
        Decision Tree and KNN are tuned using GridSearchCV.
        Macro F1-Score is used as the main criterion when selecting
        the best overall model.
        """
    )

    # --------------------------------------------------------
    # PARAMETERS
    # --------------------------------------------------------

    st.subheader("Model Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Logistic Regression")
        st.caption("Baseline")
        st.write(
            {
                "C": 1.0,
                "solver": "lbfgs",
                "penalty": "l2",
                "max_iter": 3000
            }
        )
        st.write("Tuning: Fixed parameters")

    with col2:
        st.markdown("#### Decision Tree")
        st.caption("GridSearchCV tuned")
        st.write(
            {
                "criterion": "entropy",
                "max_depth": 10,
                "min_samples_leaf": 3,
                "min_samples_split": 5
            }
        )
        st.write("CV Macro F1: 90.98%")

    with col3:
        st.markdown("#### K-Nearest Neighbors")
        st.caption("GridSearchCV tuned")
        st.write(
            {
                "metric": "manhattan",
                "n_neighbors": 5,
                "p": 1,
                "weights": "distance"
            }
        )
        st.write("CV Score: 86.96%")

    st.divider()

    # --------------------------------------------------------
    # PERFORMANCE DATAFRAME
    # --------------------------------------------------------

    st.subheader("Overall Performance Comparison")

    performance_df = pd.DataFrame(
        {
            "Model": [
                "Decision Tree",
                "Logistic Regression",
                "K-Nearest Neighbors"
            ],
            "Accuracy (%)": [
                93.81,
                86.38,
                85.14
            ],
            "Macro Precision (%)": [
                91.74,
                83.44,
                80.69
            ],
            "Macro Recall (%)": [
                92.01,
                83.20,
                81.10
            ],
            "Macro F1-Score (%)": [
                91.82,
                83.08,
                80.34
            ]
        }
    )

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # GROUPED BAR CHART - SAME STYLE AS NOTEBOOK
    # --------------------------------------------------------

    models = performance_df["Model"].tolist()

    accuracy = performance_df["Accuracy (%)"].tolist()
    precision = performance_df["Macro Precision (%)"].tolist()
    recall = performance_df["Macro Recall (%)"].tolist()
    f1_score = performance_df["Macro F1-Score (%)"].tolist()

    x = list(range(len(models)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(13, 6))

    bars1 = ax.bar(
        [i - 1.5 * width for i in x],
        accuracy,
        width,
        label="Accuracy"
    )

    bars2 = ax.bar(
        [i - 0.5 * width for i in x],
        precision,
        width,
        label="Macro Precision"
    )

    bars3 = ax.bar(
        [i + 0.5 * width for i in x],
        recall,
        width,
        label="Macro Recall"
    )

    bars4 = ax.bar(
        [i + 1.5 * width for i in x],
        f1_score,
        width,
        label="Macro F1-Score"
    )

    # Add percentage labels above every bar
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            value = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.6,
                f"{value:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )

    ax.set_title(
        "Overall Performance Comparison of Machine Learning Models",
        fontsize=14
    )

    ax.set_xlabel(
        "Machine Learning Model"
    )

    ax.set_ylabel(
        "Performance (%)"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(models)

    ax.set_ylim(0, 100)

    ax.legend(
        title="Evaluation Metrics"
    )

    fig.tight_layout()

    st.pyplot(fig)

    plt.close(fig)

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    st.subheader("Best Overall Model")

    st.success("🏆 Decision Tree")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "93.81%")
    col2.metric("Macro Precision", "91.74%")
    col3.metric("Macro Recall", "92.01%")
    col4.metric("Macro F1", "91.82%")

    st.write(
        """
        **Why Decision Tree was selected:**  
        It achieved the highest test-set Accuracy and Macro F1-Score
        among the three evaluated models. The regularized tree also uses
        `min_samples_leaf=3` and `min_samples_split=5` to reduce overly
        small leaf nodes and make its probability outputs less extreme.
        """
    )


# ============================================================
# LIVE PREDICTION
# ============================================================

elif page == "Live Prediction":

    st.title("🔍 Live Obesity Level Prediction")

    st.write(
        """
        Enter the person's physical, dietary, and lifestyle information
        below. The saved Decision Tree model will process the input using
        the same feature structure and scaler used during training.
        """
    )

    st.info(
        f"Model currently loaded: {type(model).__name__}"
    )

    # --------------------------------------------------------
    # BASIC INFORMATION
    # --------------------------------------------------------

    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.number_input(
            "Age",
            min_value=10.0,
            max_value=100.0,
            value=25.0,
            step=1.0
        )

        height = st.number_input(
            "Height (metres)",
            min_value=1.20,
            max_value=2.20,
            value=1.70,
            step=0.01
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=250.0,
            value=70.0,
            step=1.0
        )

    with col2:
        family_history = st.selectbox(
            "Family history of overweight?",
            ["yes", "no"]
        )

        favc = st.selectbox(
            "Frequently consume high-calorie food?",
            ["yes", "no"]
        )

        smoke = st.selectbox(
            "Do you smoke?",
            ["yes", "no"]
        )

        scc = st.selectbox(
            "Do you monitor calorie intake?",
            ["yes", "no"]
        )

    bmi = weight / (height ** 2)

    st.metric(
        "Calculated BMI",
        f"{bmi:.2f}"
    )

    # --------------------------------------------------------
    # EATING HABITS
    # --------------------------------------------------------

    st.subheader("Eating Habits")

    col1, col2 = st.columns(2)

    with col1:
        fcvc = st.slider(
            "Vegetable consumption frequency (FCVC)",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.1
        )

        ncp = st.slider(
            "Number of main meals per day (NCP)",
            min_value=1.0,
            max_value=4.0,
            value=3.0,
            step=0.1
        )

        caec = st.selectbox(
            "Food consumption between meals (CAEC)",
            [
                "no",
                "Sometimes",
                "Frequently",
                "Always"
            ]
        )

    with col2:
        ch2o = st.slider(
            "Daily water consumption (CH2O)",
            min_value=1.0,
            max_value=3.0,
            value=2.0,
            step=0.1
        )

        calc = st.selectbox(
            "Alcohol consumption (CALC)",
            [
                "no",
                "Sometimes",
                "Frequently",
                "Always"
            ]
        )

    # --------------------------------------------------------
    # LIFESTYLE
    # --------------------------------------------------------

    st.subheader("Lifestyle")

    col1, col2 = st.columns(2)

    with col1:
        faf = st.slider(
            "Physical activity frequency (FAF)",
            min_value=0.0,
            max_value=3.0,
            value=1.0,
            step=0.1
        )

        tue = st.slider(
            "Technology usage time (TUE)",
            min_value=0.0,
            max_value=2.0,
            value=1.0,
            step=0.1
        )

    with col2:
        mtrans = st.selectbox(
            "Main mode of transportation (MTRANS)",
            [
                "Automobile",
                "Bike",
                "Motorbike",
                "Public_Transportation",
                "Walking"
            ]
        )

    st.divider()

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button(
        "Predict Obesity Level",
        type="primary",
        use_container_width=True
    ):

        # Create exact encoded structure used during training
        input_encoded = pd.DataFrame(
            0.0,
            index=[0],
            columns=feature_columns
        )

        # Numerical columns
        numeric_values = {
            "Age": age,
            "Height": height,
            "Weight": weight,
            "FCVC": fcvc,
            "NCP": ncp,
            "CH2O": ch2o,
            "FAF": faf,
            "TUE": tue
        }

        for column, value in numeric_values.items():
            if column in input_encoded.columns:
                input_encoded.loc[0, column] = value

        # Categorical inputs
        categorical_values = {
            "Gender": gender,
            "family_history_with_overweight": family_history,
            "FAVC": favc,
            "CAEC": caec,
            "SMOKE": smoke,
            "SCC": scc,
            "CALC": calc,
            "MTRANS": mtrans
        }

        # Manual one-hot encoding.
        # If a category is the drop_first baseline, no dummy column is set,
        # which correctly represents that baseline category.
        for feature, value in categorical_values.items():
            dummy_column = f"{feature}_{value}"

            if dummy_column in input_encoded.columns:
                input_encoded.loc[0, dummy_column] = 1.0

        # Scale input using the saved training scaler
        input_scaled = scaler.transform(input_encoded)

        # Prediction
        prediction = model.predict(input_scaled)

        predicted_class = label_encoder.inverse_transform(
            prediction.astype(int)
        )[0]

        st.success(
            f"Predicted Obesity Level: **{predicted_class}**"
        )

        # ----------------------------------------------------
        # MODEL PROBABILITY
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_scaled
            )[0]

            highest_probability = probabilities.max() * 100

            st.metric(
                "Model Probability",
                f"{highest_probability:.2f}%"
            )

            st.caption(
                "This is the model's prediction probability. "
                "It is not a guarantee of real-world or medical certainty."
            )

            class_labels = label_encoder.inverse_transform(
                model.classes_.astype(int)
            )

            probability_df = pd.DataFrame(
                {
                    "Obesity Level": class_labels,
                    "Probability (%)": probabilities * 100
                }
            )

            probability_df["Probability (%)"] = (
                probability_df["Probability (%)"]
                .round(2)
            )

            probability_df = (
                probability_df
                .sort_values(
                    "Probability (%)",
                    ascending=False
                )
                .reset_index(drop=True)
            )

            st.subheader("Class Probability")

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )

            st.bar_chart(
                probability_df.set_index(
                    "Obesity Level"
                )["Probability (%)"]
            )

        with st.expander("Show processed model input"):
            st.dataframe(
                input_encoded,
                use_container_width=True
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BMDS2003 Data Science — Obesity Level Classification Prototype"
)
