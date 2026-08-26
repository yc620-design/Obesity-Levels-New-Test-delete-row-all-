
import streamlit as st
import pandas as pd
import joblib
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
        individuals into seven obesity levels based on physical,
        dietary, and lifestyle attributes.

        **Method 1** is used during data preparation to reduce highly
        interpolated observations before model training.
        """
    )

    st.info(
        "The system compares Decision Tree, Logistic Regression, and "
        "K-Nearest Neighbors, then deploys the best-performing model."
    )

    st.subheader("Dataset Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows Before Cleaning", "2,111")
    col2.metric("After Duplicate Removal", "2,087")
    col3.metric("Rows After Method 1", "877")
    col4.metric("Target Classes", "7")

    removed_total = 2111 - 877
    st.caption(
        f"Total rows removed from the original dataset: {removed_total:,}"
    )

    st.subheader("Class Distribution After Method 1")

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
                101,
                282,
                102,
                124,
                152,
                18,
                98
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

    st.title("🧹 Data Preparation — Method 1")

    st.write(
        """
        Method 1 identifies highly interpolated observations using five
        ordinal survey variables: **FCVC, NCP, CH2O, FAF, and TUE**.

        A row is considered highly interpolated when at least **3 of the
        5 variables contain non-integer values**. These rows are removed
        before encoding and model training.
        """
    )

    st.subheader("Before and After Cleaning")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Original Rows", "2,111")
    col2.metric("After Duplicates", "2,087")
    col3.metric("Rows Removed by Method 1", "1,210")
    col4.metric("Final Rows", "877")

    cleaning_df = pd.DataFrame(
        {
            "Stage": [
                "Original Dataset",
                "After Duplicate Removal",
                "After Method 1 Filtering"
            ],
            "Rows": [
                2111,
                2087,
                877
            ]
        }
    )

    st.subheader("Cleaning Result")

    st.dataframe(
        cleaning_df,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        cleaning_df.set_index("Stage")["Rows"]
    )

    st.subheader("Method 1 Rule")

    st.code(
        """
Interpolated_Count < 3  -> Keep row
Interpolated_Count >= 3 -> Remove row
        """.strip()
    )

    example_df = pd.DataFrame(
        {
            "Feature": [
                "FCVC",
                "NCP",
                "CH2O",
                "FAF",
                "TUE"
            ],
            "Example Value": [
                2.184707,
                3.000000,
                1.978631,
                0.838957,
                1.000000
            ],
            "Interpolated?": [
                "Yes",
                "No",
                "Yes",
                "Yes",
                "No"
            ]
        }
    )

    st.dataframe(
        example_df,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "The example above has 3 interpolated values, so Method 1 "
        "classifies the entire row as highly interpolated and removes it."
    )

    st.subheader("Train / Test Split")

    col1, col2, col3 = st.columns(3)

    col1.metric("Training Samples", "701")
    col2.metric("Testing Samples", "176")
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
                "6",
                "7",
                "8"
            ],
            "Process": [
                "Missing value check",
                "Duplicate removal",
                "Invalid-value check",
                "Interpolated-value detection",
                "Method 1 row filtering",
                "Target and categorical encoding",
                "80/20 stratified split",
                "Feature standardization"
            ],
            "Method": [
                "Check null values",
                "drop_duplicates()",
                "Check Age, Height and Weight > 0",
                "Check FCVC, NCP, CH2O, FAF and TUE",
                "Remove rows where Interpolated_Count >= 3",
                "LabelEncoder + one-hot encoding",
                "random_state=42, stratify=y",
                "StandardScaler fitted on training data only"
            ]
        }
    )

    st.dataframe(
        preprocessing_df,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "The scaler is fitted only on the training set and then used "
        "to transform both the test data and new Streamlit input."
    )

    with st.expander("Encoded feature columns used by the model"):
        st.write(feature_columns)


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.title("🤖 Model Comparison — Method 1")

    st.write(
        """
        Logistic Regression is used as the baseline model.
        Decision Tree and KNN are tuned using GridSearchCV.
        The models are evaluated using Accuracy, Macro Precision,
        Macro Recall, and Macro F1-Score.
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
                "min_samples_leaf": 5,
                "min_samples_split": 5
            }
        )
        st.write("CV Macro F1: 84.21%")

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
        st.write("CV Accuracy: 72.05%")

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
                89.20,
                81.25,
                78.98
            ],
            "Macro Precision (%)": [
                90.44,
                74.32,
                75.66
            ],
            "Macro Recall (%)": [
                83.38,
                72.27,
                70.79
            ],
            "Macro F1-Score (%)": [
                85.58,
                72.72,
                72.32
            ]
        }
    )

    st.dataframe(
        performance_df,
        use_container_width=True,
        hide_index=True
    )

    chart_df = performance_df.set_index("Model")

    st.bar_chart(
        chart_df[
            [
                "Accuracy (%)",
                "Macro Precision (%)",
                "Macro Recall (%)",
                "Macro F1-Score (%)"
            ]
        ]
    )

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    st.subheader("Best Overall Model")

    st.success("🏆 Decision Tree")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "89.20%")
    col2.metric("Macro Precision", "90.44%")
    col3.metric("Macro Recall", "83.38%")
    col4.metric("Macro F1", "85.58%")

    st.write(
        """
        **Why Decision Tree was selected:**  
        Decision Tree achieved the highest Macro F1-Score and Accuracy
        among the three evaluated models after Method 1 data preparation.
        Its tuned configuration also limits very small leaf nodes through
        `min_samples_leaf=5` and `min_samples_split=5`.
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
        fcvc = st.selectbox(
            "Vegetable consumption frequency (FCVC)",
            [1.0, 2.0, 3.0],
            index=1
        )

        ncp = st.selectbox(
            "Number of main meals per day (NCP)",
            [1.0, 2.0, 3.0, 4.0],
            index=2
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
        ch2o = st.selectbox(
            "Daily water consumption (CH2O)",
            [1.0, 2.0, 3.0],
            index=1
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
        faf = st.selectbox(
            "Physical activity frequency (FAF)",
            [0.0, 1.0, 2.0, 3.0],
            index=1
        )

        tue = st.selectbox(
            "Technology usage time (TUE)",
            [0.0, 1.0, 2.0],
            index=1
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
