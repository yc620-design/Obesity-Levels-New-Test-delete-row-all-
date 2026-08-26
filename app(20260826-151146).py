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
# LOAD TRAINED MODEL + PREPROCESSING OBJECTS
# ============================================================

@st.cache_resource
def load_artifacts():
    base_dir = Path(__file__).resolve().parent

    model = joblib.load(base_dir / "best_obesity_model.pkl")
    scaler = joblib.load(base_dir / "scaler.pkl")
    label_encoder = joblib.load(base_dir / "label_encoder.pkl")
    feature_columns = joblib.load(base_dir / "feature_columns.pkl")

    return model, scaler, label_encoder, feature_columns


try:
    model, scaler, label_encoder, feature_columns = load_artifacts()
except Exception as e:
    st.error(
        "Unable to load the trained model files. "
        "Make sure best_obesity_model.pkl, scaler.pkl, "
        "label_encoder.pkl and feature_columns.pkl are in the same folder as app.py."
    )
    st.code(str(e))
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Obesity Classification")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Data Preparation",
        "Model Comparison",
        "Live Prediction"
    ]
)

st.sidebar.divider()
st.sidebar.caption(f"Loaded model: {type(model).__name__}")


# ============================================================
# PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":

    st.title("📊 Obesity Level Classification")

    st.write(
        """
        This project classifies an individual into one of seven obesity
        level categories using physical, dietary and lifestyle information.

        Three machine learning models were evaluated:
        Decision Tree, Logistic Regression and K-Nearest Neighbors.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Original Records", "2,111")
    col2.metric("After Method 1", "877")
    col3.metric("Encoded Features", "23")
    col4.metric("Target Classes", "7")

    st.subheader("Target Classes")

    target_classes = pd.DataFrame(
        {
            "Obesity Level": [
                "Insufficient_Weight",
                "Normal_Weight",
                "Obesity_Type_I",
                "Obesity_Type_II",
                "Obesity_Type_III",
                "Overweight_Level_I",
                "Overweight_Level_II"
            ]
        }
    )

    st.dataframe(
        target_classes,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "This prototype is for academic demonstration only and is not a medical diagnosis."
    )


# ============================================================
# DATA PREPARATION - METHOD 1
# ============================================================

elif page == "Data Preparation":

    st.title("🧹 Data Preparation — Method 1")

    st.write(
        """
        Method 1 removes highly interpolated observations before model
        training. Five ordinal survey features are checked:
        **FCVC, NCP, CH2O, FAF and TUE**.
        """
    )

    st.subheader("Method 1 Rule")

    st.code(
        """
Interpolated_Count < 3  -> Keep the row
Interpolated_Count >= 3 -> Remove the row
        """.strip()
    )

    st.write(
        """
        A value such as **2.000000** is treated as an integer and is not
        considered interpolated. Values such as **2.184707**, **1.978631**
        or **0.838957** are treated as interpolated values.

        A row is removed only when at least **3 out of the 5 selected
        ordinal variables** contain interpolated values.
        """
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Original Records", "2,111")
    col2.metric("After Duplicate Removal", "2,087")
    col3.metric("After Method 1 Filtering", "877")

    st.subheader("Preprocessing Flow")

    preparation_df = pd.DataFrame(
        {
            "Step": [
                1, 2, 3, 4, 5, 6, 7, 8
            ],
            "Process": [
                "Missing value check",
                "Remove exact duplicates",
                "Check invalid physical values",
                "Detect interpolated ordinal values",
                "Remove highly interpolated rows",
                "Encode categorical variables",
                "80/20 stratified train-test split",
                "Standardization"
            ],
            "Method": [
                "Check null values",
                "drop_duplicates()",
                "Age, Height and Weight must be > 0",
                "Check FCVC, NCP, CH2O, FAF and TUE",
                "Remove rows with Interpolated_Count >= 3",
                "LabelEncoder + One-Hot Encoding",
                "random_state=42, stratify=y",
                "StandardScaler fitted on training data only"
            ]
        }
    )

    st.dataframe(
        preparation_df,
        use_container_width=True,
        hide_index=True
    )

    st.success(
        "The scaler is fitted only on the training set to avoid data leakage."
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
        The following results are from the Method 1 dataset preparation
        used in the current notebook.
        """
    )

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

    st.subheader("Overall Performance Comparison")

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

    st.subheader("Best Model")

    st.success("🏆 Decision Tree")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Accuracy", "89.20%")
    col2.metric("Macro Precision", "90.44%")
    col3.metric("Macro Recall", "83.38%")
    col4.metric("Macro F1", "85.58%")

    st.subheader("Model Configuration")

    config_df = pd.DataFrame(
        {
            "Model": [
                "Decision Tree",
                "Logistic Regression",
                "K-Nearest Neighbors"
            ],
            "Configuration": [
                "criterion=entropy, max_depth=10, min_samples_leaf=5, min_samples_split=5",
                "C=1.0, solver=lbfgs, penalty=l2, max_iter=3000",
                "metric=manhattan, n_neighbors=5, p=1, weights=distance"
            ]
        }
    )

    st.dataframe(
        config_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# LIVE PREDICTION
# ============================================================

elif page == "Live Prediction":

    st.title("🔍 Live Obesity Level Prediction")

    st.write(
        """
        Enter the person's physical, dietary and lifestyle information.
        The app applies the same encoded feature structure and scaler
        used during model training.
        """
    )

    st.info(
        "For Method 1 consistency, FCVC, NCP, CH2O, FAF and TUE are "
        "entered using discrete survey values."
    )

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    st.subheader("Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        age = st.number_input(
            "Age",
            min_value=14.0,
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
            step=0.5
        )

    with col2:
        family_history = st.selectbox(
            "Family history of overweight?",
            ["no", "yes"]
        )

        favc = st.selectbox(
            "Frequently consume high-calorie food?",
            ["no", "yes"]
        )

        smoke = st.selectbox(
            "Do you smoke?",
            ["no", "yes"]
        )

        scc = st.selectbox(
            "Do you monitor calorie intake?",
            ["no", "yes"]
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
                "Always",
                "Frequently",
                "Sometimes",
                "no"
            ],
            index=2
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
                "Always",
                "Frequently",
                "Sometimes",
                "no"
            ],
            index=2
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
            ],
            index=3
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

        # Create the exact encoded feature structure saved from training.
        input_encoded = pd.DataFrame(
            0.0,
            index=[0],
            columns=feature_columns
        )

        # Numerical / ordinal features
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

        # Categorical variables
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
        # Baseline categories from drop_first=True remain all-zero.
        for feature, value in categorical_values.items():
            dummy_column = f"{feature}_{value}"

            if dummy_column in input_encoded.columns:
                input_encoded.loc[0, dummy_column] = 1.0

        # Apply the scaler fitted on the training data.
        input_scaled = scaler.transform(input_encoded)

        # Predict encoded target.
        prediction = model.predict(input_scaled)

        # Convert encoded target back to obesity label.
        predicted_class = label_encoder.inverse_transform(
            prediction.astype(int)
        )[0]

        st.success(
            f"Predicted Obesity Level: **{predicted_class}**"
        )

        # ----------------------------------------------------
        # PREDICTION PROBABILITIES
        # ----------------------------------------------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_scaled
            )[0]

            model_classes = model.classes_.astype(int)

            class_labels = label_encoder.inverse_transform(
                model_classes
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

            highest_probability = (
                probability_df.loc[0, "Probability (%)"]
            )

            st.metric(
                "Highest Model Probability",
                f"{highest_probability:.2f}%"
            )

            st.caption(
                "This probability is a model output and should not be interpreted "
                "as medical certainty."
            )

            st.subheader("Class Probabilities")

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
