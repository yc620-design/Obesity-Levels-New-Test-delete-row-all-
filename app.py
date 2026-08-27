import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

BASE_DIR = Path(__file__).resolve().parent

# ============================================================
# LOAD MODEL FILES
# ============================================================

@st.cache_resource
def load_model_files():
    model = joblib.load(BASE_DIR / "best_obesity_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")
    return model, scaler, label_encoder, feature_columns


model = None
scaler = None
label_encoder = None
feature_columns = None
model_error = None

try:
    model, scaler, label_encoder, feature_columns = load_model_files()
except Exception as e:
    model_error = str(e)


# ============================================================
# LOAD ORIGINAL DATASET FOR DATA ANALYSIS
# ============================================================

@st.cache_data
def load_dataset():
    possible_files = [
        BASE_DIR / "ObesityDataSet_raw_and_data_sinthetic.csv",
        BASE_DIR / "ObesityDataSet_raw_and_data_sinthetic(1).csv"
    ]

    for file_path in possible_files:
        if file_path.exists():
            return pd.read_csv(file_path), file_path.name

    return None, None


obe, dataset_filename = load_dataset()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("Obesity Classification")

page = st.sidebar.radio(
    "Presentation Navigation",
    [
        "Project Overview",
        "Data Analysis",
        "Data Preparation",
        "Model Comparison",
        "Live Prediction"
    ]
)

st.sidebar.divider()

if model is not None:
    st.sidebar.caption(
        f"Loaded model: {type(model).__name__}"
    )
else:
    st.sidebar.warning(
        "Model files are not loaded. "
        "Data Analysis can still be viewed if the CSV is available."
    )


# ============================================================
# PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":

    st.title("📊 Obesity Level Classification")

    st.write(
        """
        This project applies machine learning techniques to classify
        individuals into seven obesity levels using physical, dietary,
        and lifestyle attributes.
        """
    )

    st.info(
        "Three machine learning models are compared: Decision Tree, "
        "Logistic Regression, and K-Nearest Neighbors."
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Original Records", "2,111")
    col2.metric("After Data Preparation", "877")
    col3.metric("Encoded Features", "23")
    col4.metric("Target Classes", "7")

    st.subheader("Class Distribution After Data Preparation")

    class_df = pd.DataFrame(
        {
            "Obesity Level": [
                "Normal_Weight",
                "Obesity_Type_I",
                "Overweight_Level_II",
                "Overweight_Level_I",
                "Insufficient_Weight",
                "Obesity_Type_III",
                "Obesity_Type_II"
            ],
            "Records": [
                282,
                152,
                124,
                102,
                101,
                98,
                18
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
# DATA ANALYSIS
# ============================================================

elif page == "Data Analysis":

    st.title("📈 Data Analysis")

    st.write(
        """
        The following visualisations are based on the original dataset
        before model training and follow the analysis used in the notebook.
        """
    )

    if obe is None:
        st.error(
            "Dataset CSV not found. Upload "
            "`ObesityDataSet_raw_and_data_sinthetic.csv` "
            "to the same GitHub folder as `app.py`."
        )
        st.stop()

    st.caption(
        f"Dataset loaded: {dataset_filename} | "
        f"{obe.shape[0]:,} rows × {obe.shape[1]} columns"
    )

    # --------------------------------------------------------
    # GRAPH 1 - OBESITY LEVEL DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("1. Distribution of Obesity Levels")

    fig1, ax1 = plt.subplots(figsize=(10, 5))

    sns.countplot(
        data=obe,
        x="NObeyesdad",
        ax=ax1
    )

    ax1.set_title("Distribution of Obesity Levels")
    ax1.set_xlabel("Obesity Level")
    ax1.set_ylabel("Count")
    ax1.tick_params(axis="x", rotation=45)

    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.caption(
        "This graph shows the number of records in each obesity category."
    )

    # --------------------------------------------------------
    # GRAPH 2 - EATING BETWEEN MEALS
    # --------------------------------------------------------

    st.subheader(
        "2. Eating Between Meals (CAEC) Broken Down by Obesity Level"
    )

    snack_order = [
        value for value in [
            "no",
            "Sometimes",
            "Frequently",
            "Always"
        ]
        if value in obe["CAEC"].unique()
    ]

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    sns.countplot(
        data=obe,
        x="CAEC",
        hue="NObeyesdad",
        order=snack_order,
        palette="viridis",
        ax=ax2
    )

    ax2.set_title(
        "Eating Between Meals (CAEC) Broken Down by Obesity Level",
        fontsize=14,
        fontweight="bold"
    )

    ax2.set_xlabel(
        "Frequency of Snacking Between Meals",
        fontsize=12
    )

    ax2.set_ylabel(
        "Number of Individuals",
        fontsize=12
    )

    ax2.legend(
        title="Obesity Level",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )

    ax2.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.caption(
        "This graph compares snacking frequency across the seven obesity levels."
    )

    # --------------------------------------------------------
    # GRAPH 3 - FAF VS TUE
    # --------------------------------------------------------

    st.subheader(
        "3. Average Physical Activity (FAF) vs Technology Time (TUE)"
    )

    avg_habits = (
        obe.groupby("NObeyesdad")[["FAF", "TUE"]]
        .mean()
        .reset_index()
    )

    avg_habits_melted = avg_habits.melt(
        id_vars="NObeyesdad",
        var_name="Habit",
        value_name="Average Score"
    )

    fig3, ax3 = plt.subplots(figsize=(12, 6))

    sns.pointplot(
        data=avg_habits_melted,
        x="NObeyesdad",
        y="Average Score",
        hue="Habit",
        markers=["o", "s"],
        linestyles=["-", "--"],
        palette="Dark2",
        ax=ax3
    )

    ax3.set_title(
        "Average Physical Activity (FAF) vs. Technology Time (TUE) "
        "by Weight Category",
        fontsize=14,
        fontweight="bold"
    )

    ax3.set_xlabel(
        "Weight Category (NObeyesdad)",
        fontsize=12
    )

    ax3.set_ylabel(
        "Average Feature Score",
        fontsize=12
    )

    ax3.tick_params(
        axis="x",
        rotation=30
    )

    ax3.grid(
        True,
        linestyle="--",
        alpha=0.5
    )

    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.caption(
        "FAF represents physical activity frequency, while TUE represents "
        "technology-use time."
    )

    # --------------------------------------------------------
    # GRAPH 4 - HEIGHT VS WEIGHT
    # --------------------------------------------------------

    st.subheader("4. Scatterplot: Height vs Weight")

    b, m = np.polynomial.polynomial.polyfit(
        obe["Height"],
        obe["Weight"],
        1
    )

    fig4, ax4 = plt.subplots(figsize=(9, 6))

    ax4.plot(
        obe["Height"],
        obe["Weight"],
        "."
    )

    ax4.plot(
        obe["Height"],
        b + m * obe["Height"],
        "-"
    )

    ax4.set_xlabel("Height")
    ax4.set_ylabel("Weight")
    ax4.set_title("Scatterplot: Height vs Weight")

    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    st.caption(
        "The scatterplot is used to examine the relationship between "
        "height and body weight."
    )

    # --------------------------------------------------------
    # GRAPH 5 - CORRELATION HEATMAP
    # --------------------------------------------------------

    st.subheader("5. Correlation Heatmap")

    corr = obe.corr(
        numeric_only=True
    )

    fig5, ax5 = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        annot=True,
        fmt=".2f",
        cmap="rocket",
        ax=ax5
    )

    ax5.set_title(
        "Correlation Heatmap for Obesity Dataset"
    )

    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    st.caption(
        "The heatmap shows the strength and direction of relationships "
        "among the numerical variables."
    )


# ============================================================
# DATA PREPARATION
# ============================================================

elif page == "Data Preparation":

    st.title("🧹 Data Preparation")

    st.write(
        """
        The notebook removes exact duplicate records, checks invalid
        physical values, and filters highly interpolated questionnaire
        records before encoding and model training.
        """
    )

    st.subheader("Rows Before and After Cleaning")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Original Rows", "2,111")
    col2.metric("After Duplicate Removal", "2,087")
    col3.metric("Rows Removed by Filtering", "1,210")
    col4.metric("Final Rows", "877")

    cleaning_df = pd.DataFrame(
        {
            "Stage": [
                "Original Dataset",
                "After Duplicate Removal",
                "After Interpolation Filtering"
            ],
            "Rows": [
                2111,
                2087,
                877
            ]
        }
    )

    st.dataframe(
        cleaning_df,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        cleaning_df.set_index("Stage")["Rows"]
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
                "Missing-value check",
                "Remove exact duplicates",
                "Invalid-value check",
                "Detect interpolated ordinal values",
                "Filter highly interpolated records",
                "Target and categorical encoding",
                "80/20 stratified split",
                "Feature standardization"
            ],
            "Method": [
                "No missing values found",
                "24 duplicate rows removed",
                "Age, Height and Weight checked for invalid values",
                "FCVC, NCP, CH2O, FAF and TUE checked",
                "Remove rows with Interpolated_Count >= 3",
                "LabelEncoder + one-hot encoding",
                "random_state=42 and stratify=y",
                "StandardScaler fitted on training data only"
            ]
        }
    )

    st.dataframe(
        preprocessing_df,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "Filtering rule: if at least 3 of the 5 selected ordinal survey "
        "variables contain interpolated non-integer values, the row is removed."
    )

    st.success(
        "The scaler is fitted only on the training set, then used to "
        "transform the test set and new Streamlit input."
    )

    if feature_columns is not None:
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
        Decision Tree is tuned using Macro F1-Score, while KNN is tuned
        using accuracy in GridSearchCV. The final models are evaluated
        using Accuracy, Macro Precision, Macro Recall, and Macro F1-Score.
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
        st.write("Tuning: Fixed parameters / No GridSearchCV")

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
    # Matches the notebook evaluation output
    # --------------------------------------------------------

    st.subheader(
        "Overall Performance Comparison of Machine Learning Models"
    )

    performance_df = pd.DataFrame(
        {
            "Model": [
                "Decision Tree",
                "Logistic Regression",
                "K-Nearest Neighbors"
            ],
            "Accuracy": [
                89.2045,
                81.2500,
                78.9773
            ],
            "Macro Precision": [
                90.4424,
                74.3214,
                75.6639
            ],
            "Macro Recall": [
                83.3784,
                72.2707,
                70.7920
            ],
            "Macro F1-Score": [
                85.5800,
                72.7197,
                72.3218
            ]
        }
    )

    display_df = performance_df.copy()

    for col in [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-Score"
    ]:
        display_df[col] = (
            display_df[col]
            .map(lambda value: f"{value:.2f}%")
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # SAME STYLE AS NOTEBOOK CODING PART
    # --------------------------------------------------------

    plot_metrics = [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-Score"
    ]

    plot_df = (
        performance_df[
            ["Model"] + plot_metrics
        ]
        .set_index("Model")
    )

    fig_perf, ax_perf = plt.subplots(
        figsize=(13, 7)
    )

    plot_df.plot(
        kind="bar",
        width=0.75,
        ax=ax_perf
    )

    ax_perf.set_title(
        "Overall Performance Comparison of Machine Learning Models",
        fontsize=15,
        pad=15
    )

    ax_perf.set_xlabel(
        "Machine Learning Model",
        fontsize=12
    )

    ax_perf.set_ylabel(
        "Performance Score (%)",
        fontsize=12
    )

    ax_perf.set_ylim(
        0,
        105
    )

    ax_perf.tick_params(
        axis="x",
        rotation=0,
        labelsize=10
    )

    ax_perf.legend(
        title="Evaluation Metrics",
        loc="lower right"
    )

    for container in ax_perf.containers:
        labels = [
            f"{bar.get_height():.2f}%"
            for bar in container
        ]

        ax_perf.bar_label(
            container,
            labels=labels,
            padding=3,
            fontsize=9
        )

    fig_perf.tight_layout()
    st.pyplot(fig_perf)
    plt.close(fig_perf)

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
        It achieved the highest Macro F1-Score and Accuracy among the
        three evaluated models in the notebook.
        """
    )


# ============================================================
# LIVE PREDICTION
# ============================================================

elif page == "Live Prediction":

    st.title("🔍 Live Obesity Level Prediction")

    if model is None:
        st.error(
            "The prediction model files are not available. "
            "Upload these four files to the same GitHub folder as app.py:"
        )

        st.code(
            """
best_obesity_model.pkl
scaler.pkl
label_encoder.pkl
feature_columns.pkl
            """.strip()
        )

        if model_error:
            with st.expander("Model loading error"):
                st.code(model_error)

        st.stop()

    st.write(
        """
        Enter the person's physical, dietary, and lifestyle information
        below. The saved best model will process the input using the same
        encoded feature structure and scaler used during training.
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

        input_encoded = pd.DataFrame(
            0.0,
            index=[0],
            columns=feature_columns
        )

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

        for feature, value in categorical_values.items():
            dummy_column = f"{feature}_{value}"

            if dummy_column in input_encoded.columns:
                input_encoded.loc[0, dummy_column] = 1.0

        input_scaled = scaler.transform(
            input_encoded
        )

        prediction = model.predict(
            input_scaled
        )

        predicted_class = label_encoder.inverse_transform(
            prediction.astype(int)
        )[0]

        st.success(
            f"Predicted Obesity Level: **{predicted_class}**"
        )

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                input_scaled
            )[0]

            highest_probability = (
                probabilities.max() * 100
            )

            st.metric(
                "Model Probability",
                f"{highest_probability:.2f}%"
            )

            st.caption(
                "This is the model's prediction probability. "
                "It is not a medical diagnosis."
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
