import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import joblib
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Obesity Level Classification",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3 {
            letter-spacing: -0.02em;
        }
        [data-testid="stMetric"] {
            border: 1px solid rgba(128,128,128,0.25);
            border-radius: 12px;
            padding: 12px;
        }
        .small-note {
            font-size: 0.92rem;
            opacity: 0.82;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSTANTS FROM FINAL NOTEBOOK
# ============================================================

NOTEBOOK_RESULTS = pd.DataFrame(
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

CLASS_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
    "Overweight_Level_I",
    "Overweight_Level_II"
]

ORDINAL_FEATURES = [
    "FCVC",
    "NCP",
    "CH2O",
    "FAF",
    "TUE"
]


# ============================================================
# LOAD DATA / MODEL FILES
# ============================================================

@st.cache_data
def load_raw_dataset():
    candidates = [
        BASE_DIR / "ObesityDataSet_raw_and_data_sinthetic.csv",
        BASE_DIR / "ObesityDataSet_raw_and_data_sinthetic(1).csv"
    ]

    for path in candidates:
        if path.exists():
            return pd.read_csv(path), path.name

    return None, None


@st.cache_resource
def load_model_files():
    model = joblib.load(BASE_DIR / "best_obesity_model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    label_encoder = joblib.load(BASE_DIR / "label_encoder.pkl")
    feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")

    return model, scaler, label_encoder, feature_columns


raw_df, dataset_name = load_raw_dataset()

model = None
scaler = None
label_encoder = None
feature_columns = None
model_error = None

try:
    model, scaler, label_encoder, feature_columns = load_model_files()
except Exception as exc:
    model_error = str(exc)


# ============================================================
# DATA PREPARATION — MATCHES FINAL NOTEBOOK
# ============================================================

@st.cache_data
def prepare_dataset(df):
    work = df.copy()

    original_rows = len(work)
    missing_values = int(work.isnull().sum().sum())
    duplicate_rows = int(work.duplicated().sum())

    work = (
        work
        .drop_duplicates()
        .reset_index(drop=True)
        .copy()
    )

    after_duplicates = len(work)

    invalid_mask = (
        (work["Age"] <= 0)
        | (work["Height"] <= 0)
        | (work["Weight"] <= 0)
    )

    invalid_rows = int(invalid_mask.sum())

    work = (
        work.loc[~invalid_mask]
        .reset_index(drop=True)
        .copy()
    )

    interpolation_count = np.zeros(
        len(work),
        dtype=int
    )

    for column in ORDINAL_FEATURES:
        is_interpolated = ~np.isclose(
            work[column],
            np.round(work[column])
        )

        interpolation_count += (
            is_interpolated.astype(int)
        )

    work["Interpolated_Count"] = interpolation_count

    rows_removed_by_rule = int(
        (work["Interpolated_Count"] >= 3).sum()
    )

    work = (
        work.loc[
            work["Interpolated_Count"] < 3
        ]
        .drop(columns=["Interpolated_Count"])
        .reset_index(drop=True)
        .copy()
    )

    stats = {
        "original_rows": original_rows,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "after_duplicates": after_duplicates,
        "invalid_rows": invalid_rows,
        "rows_removed_by_rule": rows_removed_by_rule,
        "final_rows": len(work)
    }

    return work, stats


if raw_df is not None:
    cleaned_df, cleaning_stats = prepare_dataset(raw_df)
else:
    cleaned_df = None
    cleaning_stats = None


# ============================================================
# RECREATE FINAL TEST SET FOR BEST-MODEL EVALUATION
# ============================================================

def recreate_test_set(clean_df):
    if (
        clean_df is None
        or label_encoder is None
        or feature_columns is None
        or scaler is None
    ):
        return None, None

    temp = clean_df.copy()

    y = label_encoder.transform(
        temp["NObeyesdad"]
    )

    X = temp.drop(
        columns=["NObeyesdad"]
    )

    X_encoded = pd.get_dummies(
        X,
        drop_first=True,
        dtype=int
    )

    X_encoded = X_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    _, X_test, _, y_test = train_test_split(
        X_encoded,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    return X_test_scaled, y_test


def percent(value):
    return f"{value:.2f}%"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📊 Obesity Classification")

page = st.sidebar.radio(
    "Navigation",
    [
        "Project Overview",
        "Data Analysis",
        "Data Preparation",
        "Model Evaluation",
        "Live Prediction"
    ]
)

st.sidebar.divider()

if raw_df is not None:
    st.sidebar.success(
        f"Dataset loaded: {len(raw_df):,} rows"
    )
else:
    st.sidebar.error("Dataset CSV not found")

if model is not None:
    st.sidebar.success(
        f"Model loaded: {type(model).__name__}"
    )
else:
    st.sidebar.warning("Prediction model not loaded")


# ============================================================
# PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":

    st.title("Obesity Level Classification")

    st.write(
        """
        This project applies supervised machine learning to classify
        individuals into seven obesity-level categories using physical,
        dietary and lifestyle attributes.
        """
    )

    st.info(
        "The prototype follows the final notebook pipeline and compares "
        "Decision Tree, Logistic Regression and K-Nearest Neighbors."
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Original Records",
        f"{len(raw_df):,}" if raw_df is not None else "2,111"
    )

    c2.metric(
        "Final Records",
        f"{len(cleaned_df):,}" if cleaned_df is not None else "877"
    )

    c3.metric(
        "Input Features",
        "16"
    )

    c4.metric(
        "Target Classes",
        "7"
    )

    st.subheader("Analytical Objective")

    st.write(
        """
        The analytical objective is to build and compare multiclass
        classification models and identify the model that provides the
        strongest overall balance of Accuracy, Macro Precision,
        Macro Recall and Macro F1-Score.
        """
    )

    st.subheader("Models")

    models_table = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression",
                "Decision Tree",
                "K-Nearest Neighbors"
            ],
            "Role": [
                "Baseline model",
                "Tuned model",
                "Tuned model"
            ],
            "Configuration": [
                "Fixed parameters",
                "GridSearchCV",
                "GridSearchCV"
            ]
        }
    )

    st.dataframe(
        models_table,
        use_container_width=True,
        hide_index=True
    )




# ============================================================
# DATA ANALYSIS — BASED ON NOTEBOOK CODING PART
# ============================================================

elif page == "Data Analysis":

    st.title("Data Analysis")

    if raw_df is None:
        st.error(
            "Dataset CSV is required to display the analysis graphs."
        )
        st.stop()

    # Use the same dataframe name as the notebook coding part
    obe = raw_df.copy()

    st.write(
        """
        This section follows the **Data Analysis coding part in the notebook**.
        The same variables, calculations and graph types are used so that the
        Streamlit prototype is consistent with the Python implementation.
        """
    )

    # --------------------------------------------------------
    # GRAPH 1 — DISTRIBUTION OF OBESITY LEVELS
    # Notebook:
    # sns.countplot(data=obe, x="NObeyesdad")
    # --------------------------------------------------------

    st.subheader("1. Distribution of Obesity Levels")

    fig1, ax1 = plt.subplots(
        figsize=(10, 5)
    )

    sns.countplot(
        data=obe,
        x="NObeyesdad",
        ax=ax1
    )

    ax1.tick_params(
        axis="x",
        rotation=45
    )

    ax1.set_title(
        "Distribution of Obesity Levels"
    )

    ax1.set_xlabel(
        "Obesity Level"
    )

    ax1.set_ylabel(
        "Number of Individuals"
    )

    fig1.tight_layout()

    st.pyplot(
        fig1
    )

    plt.close(
        fig1
    )

    st.caption(
        "This count plot shows how many records belong to each "
        "NObeyesdad obesity-level class."
    )

    # --------------------------------------------------------
    # GRAPH 2 — EATING BETWEEN MEALS
    # Notebook:
    # sns.countplot(data=obe, x='CAEC',
    #               hue='NObeyesdad', palette='viridis')
    # --------------------------------------------------------

    st.subheader(
        "2. Eating Between Meals (CAEC) by Obesity Level"
    )

    # The notebook uses "No", while the CSV stores this category as "no".
    # The lowercase value is used here so the same category appears correctly.
    snack_order = [
        "no",
        "Sometimes",
        "Frequently",
        "Always"
    ]

    available_order = [
        value
        for value in snack_order
        if value in obe["CAEC"].unique()
    ]

    fig2, ax2 = plt.subplots(
        figsize=(12, 6)
    )

    sns.countplot(
        data=obe,
        x="CAEC",
        hue="NObeyesdad",
        palette="viridis",
        order=available_order,
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

    st.pyplot(
        fig2
    )

    plt.close(
        fig2
    )

    st.caption(
        "CAEC represents eating between meals. The graph compares "
        "snacking-frequency categories across the obesity classes."
    )

    # --------------------------------------------------------
    # GRAPH 3 — FAF VS TUE
    # Notebook:
    # group by NObeyesdad, calculate mean FAF/TUE,
    # melt the dataframe, then use sns.pointplot()
    # --------------------------------------------------------

    st.subheader(
        "3. Average Physical Activity (FAF) vs Technology Time (TUE)"
    )

    avg_habits = (
        obe.groupby(
            "NObeyesdad"
        )[[
            "FAF",
            "TUE"
        ]]
        .mean()
        .reset_index()
    )

    avg_habits_melted = avg_habits.melt(
        id_vars="NObeyesdad",
        var_name="Habit",
        value_name="Average Score"
    )

    fig3, ax3 = plt.subplots(
        figsize=(12, 6)
    )

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

    st.pyplot(
        fig3
    )

    plt.close(
        fig3
    )

    st.caption(
        "The notebook first calculates the average FAF and TUE for "
        "each obesity class. FAF is physical activity frequency and "
        "TUE is technology-use time."
    )

    with st.expander(
        "Show Average FAF and TUE Values"
    ):
        st.dataframe(
            avg_habits.round(3),
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------------
    # GRAPH 4 — CORRELATION HEATMAP
    # Notebook:
    # corr = obe.corr(numeric_only=True)
    # sns.heatmap(... annot=True, fmt=".2f", cmap="rocket")
    # --------------------------------------------------------

    st.subheader(
        "4. Correlation Heatmap"
    )

    corr = obe.corr(
        numeric_only=True
    )

    fig5, ax5 = plt.subplots(
        figsize=(10, 8)
    )

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

    st.pyplot(
        fig5
    )

    plt.close(
        fig5
    )

    st.caption(
        "The heatmap displays pairwise correlations between the "
        "numerical variables. Values closer to +1 indicate a stronger "
        "positive relationship, while values closer to -1 indicate a "
        "stronger negative relationship."
    )


# ============================================================
# DATA PREPARATION
# ============================================================

elif page == "Data Preparation":

    st.title("Data Preparation")

    if raw_df is None:
        st.error(
            "Dataset CSV is required to recreate the notebook preprocessing."
        )
        st.stop()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Original Rows",
        f"{cleaning_stats['original_rows']:,}"
    )

    c2.metric(
        "After Duplicate Removal",
        f"{cleaning_stats['after_duplicates']:,}"
    )

    c3.metric(
        "Removed by Filtering Rule",
        f"{cleaning_stats['rows_removed_by_rule']:,}"
    )

    c4.metric(
        "Final Rows",
        f"{cleaning_stats['final_rows']:,}"
    )

    st.subheader("Cleaning Summary")

    cleaning_table = pd.DataFrame(
        {
            "Check / Transformation": [
                "Missing-value check",
                "Exact duplicate removal",
                "Invalid physical-value check",
                "Interpolated ordinal-value detection",
                "Highly interpolated record filtering",
                "Target encoding",
                "Categorical encoding",
                "Train / test split",
                "Standardisation"
            ],
            "Method": [
                f"{cleaning_stats['missing_values']} missing values found",
                f"{cleaning_stats['duplicate_rows']} exact duplicate rows removed",
                (
                    "Age, Height and Weight > 0; "
                    f"{cleaning_stats['invalid_rows']} invalid rows found"
                ),
                "Check FCVC, NCP, CH2O, FAF and TUE for non-integer values",
                "Remove records where Interpolated_Count >= 3",
                "LabelEncoder for NObeyesdad",
                "One-hot encoding with drop_first=True",
                "80% training / 20% testing, stratified, random_state=42",
                "StandardScaler fitted on training data only"
            ]
        }
    )

    st.dataframe(
        cleaning_table,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Row Retention")

    stages = pd.DataFrame(
        {
            "Stage": [
                "Original",
                "After Duplicate Removal",
                "Final Prepared Dataset"
            ],
            "Rows": [
                cleaning_stats["original_rows"],
                cleaning_stats["after_duplicates"],
                cleaning_stats["final_rows"]
            ]
        }
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        stages["Stage"],
        stages["Rows"]
    )

    ax.set_title(
        "Dataset Size Through Data Preparation"
    )

    ax.set_ylabel(
        "Number of Rows"
    )

    ax.tick_params(
        axis="x",
        rotation=15
    )

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            f"{int(bar.get_height()):,}",
            ha="center"
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Class Distribution After Preparation")

    prepared_distribution = (
        cleaned_df["NObeyesdad"]
        .value_counts()
        .rename_axis("Obesity Level")
        .reset_index(name="Records")
    )

    st.dataframe(
        prepared_distribution,
        use_container_width=True,
        hide_index=True
    )


    st.subheader("Training and Testing Data")

    train_col, test_col, split_col = st.columns(3)

    train_col.metric(
        "Training Records",
        "701"
    )

    test_col.metric(
        "Testing Records",
        "176"
    )

    split_col.metric(
        "Train / Test Split",
        "80% / 20%"
    )

    split_df = pd.DataFrame(
        {
            "Dataset": [
                "Training",
                "Testing"
            ],
            "Records": [
                701,
                176
            ]
        }
    )

    fig_split, ax_split = plt.subplots(
        figsize=(7, 4)
    )

    split_bars = ax_split.bar(
        split_df["Dataset"],
        split_df["Records"]
    )

    ax_split.set_title(
        "Training and Testing Data"
    )

    ax_split.set_ylabel(
        "Number of Records"
    )

    for bar in split_bars:
        ax_split.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 8,
            f"{int(bar.get_height())}",
            ha="center"
        )

    fig_split.tight_layout()

    st.pyplot(
        fig_split
    )

    plt.close(
        fig_split
    )

    st.caption(
        "The prepared dataset is divided into 80% training data and "
        "20% testing data using stratified sampling."
    )


# ============================================================
# MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":

    st.title("Model Evaluation")

    st.write(
        """
        All models are compared using Accuracy, Macro Precision,
        Macro Recall and Macro F1-Score. Macro averaging gives each
        obesity class equal importance when calculating Precision,
        Recall and F1.
        """
    )

    st.subheader("Model Configuration")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Logistic Regression")
        st.caption("Baseline model")
        st.json(
            {
                "C": 1.0,
                "solver": "lbfgs",
                "penalty": "l2",
                "max_iter": 3000
            }
        )
        st.write("Fixed configuration")

    with col2:
        st.markdown("#### Decision Tree")
        st.caption("GridSearchCV tuned")
        st.json(
            {
                "criterion": "entropy",
                "max_depth": 10,
                "min_samples_leaf": 5,
                "min_samples_split": 5
            }
        )
        st.write("CV Macro F1: **84.21%**")

    with col3:
        st.markdown("#### K-Nearest Neighbors")
        st.caption("GridSearchCV tuned")
        st.json(
            {
                "metric": "manhattan",
                "n_neighbors": 5,
                "p": 1,
                "weights": "distance"
            }
        )
        st.write("GridSearch scoring: **Accuracy**")
        st.write("Best CV Accuracy: **72.05%**")

    st.divider()

    st.subheader(
        "Overall Performance Comparison of Machine Learning Models"
    )

    display_results = NOTEBOOK_RESULTS.copy()

    for metric in [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1-Score"
    ]:
        display_results[metric] = (
            display_results[metric]
            .map(lambda x: f"{x:.2f}%")
        )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True
    )

    plot_df = (
        NOTEBOOK_RESULTS
        .set_index("Model")
    )

    fig, ax = plt.subplots(
        figsize=(13, 7)
    )

    plot_df.plot(
        kind="bar",
        width=0.75,
        ax=ax
    )

    ax.set_title(
        "Overall Performance Comparison of Machine Learning Models",
        fontsize=15,
        pad=15
    )

    ax.set_xlabel(
        "Machine Learning Model"
    )

    ax.set_ylabel(
        "Performance Score (%)"
    )

    ax.set_ylim(
        0,
        105
    )

    ax.tick_params(
        axis="x",
        rotation=0
    )

    ax.legend(
        title="Evaluation Metrics",
        loc="lower right"
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            labels=[
                f"{bar.get_height():.2f}%"
                for bar in container
            ],
            padding=3,
            fontsize=8
        )

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Best Overall Model")

    b1, b2, b3, b4 = st.columns(4)

    b1.metric("Accuracy", "89.20%")
    b2.metric("Macro Precision", "90.44%")
    b3.metric("Macro Recall", "83.38%")
    b4.metric("Macro F1-Score", "85.58%")

    st.success(
        "🏆 Decision Tree achieved the highest Macro F1-Score and "
        "the highest Accuracy among the three evaluated models."
    )

    st.subheader("Confusion Matrix — Best Model")

    if (
        model is not None
        and cleaned_df is not None
        and scaler is not None
        and label_encoder is not None
        and feature_columns is not None
    ):

        try:
            X_test_scaled, y_test = recreate_test_set(
                cleaned_df
            )

            best_pred = model.predict(
                X_test_scaled
            )

            dynamic_accuracy = accuracy_score(
                y_test,
                best_pred
            ) * 100

            dynamic_precision = precision_score(
                y_test,
                best_pred,
                average="macro",
                zero_division=0
            ) * 100

            dynamic_recall = recall_score(
                y_test,
                best_pred,
                average="macro",
                zero_division=0
            ) * 100

            dynamic_f1 = f1_score(
                y_test,
                best_pred,
                average="macro",
                zero_division=0
            ) * 100

            fig_cm, ax_cm = plt.subplots(
                figsize=(9, 8)
            )

            ConfusionMatrixDisplay.from_predictions(
                y_test,
                best_pred,
                display_labels=label_encoder.classes_,
                xticks_rotation=45,
                ax=ax_cm
            )

            ax_cm.set_title(
                f"Confusion Matrix - {type(model).__name__}"
            )

            fig_cm.tight_layout()
            st.pyplot(fig_cm)
            plt.close(fig_cm)

            st.caption(
                "Rows represent actual classes and columns represent "
                "predicted classes. Diagonal cells are correct predictions."
            )

            with st.expander(
                "Verify loaded best-model metrics"
            ):
                st.write(
                    {
                        "Accuracy": percent(dynamic_accuracy),
                        "Macro Precision": percent(dynamic_precision),
                        "Macro Recall": percent(dynamic_recall),
                        "Macro F1-Score": percent(dynamic_f1)
                    }
                )

        except Exception as exc:
            st.warning(
                "The confusion matrix could not be recreated from the "
                "saved model files."
            )
            st.code(str(exc))

    else:
        st.info(
            "Upload the dataset and four saved model/preprocessing files "
            "to display the best-model confusion matrix."
        )

    st.subheader("Evaluation Interpretation")

    st.write(
        """
        - **Decision Tree** provides the strongest overall performance.
        - **Macro Recall is lower than Accuracy**, indicating that some
          classes are harder to detect consistently.
        - The confusion matrix is important because it shows which
          obesity classes are being confused, rather than only reporting
          one overall score.
        """
    )




# ============================================================
# LIVE PREDICTION
# ============================================================

elif page == "Live Prediction":

    st.title("🔍 Live Obesity Level Prediction")

    # --------------------------------------------------------
    # CHECK MODEL FILES
    # --------------------------------------------------------

    if (
        model is None
        or scaler is None
        or label_encoder is None
        or feature_columns is None
    ):
        st.error(
            "Prediction files are missing. Please place these files "
            "in the same folder as app.py:"
        )

        st.code(
            """best_obesity_model.pkl
scaler.pkl
label_encoder.pkl
feature_columns.pkl"""
        )

        if model_error:
            with st.expander("Loading error"):
                st.code(model_error)

        st.stop()

    st.info(
        """
        **How to use this page**

        1. Enter your personal information.
        2. Choose your eating and lifestyle habits.
        3. The prediction appears automatically — no button is needed.
        """
    )

    # ========================================================
    # INPUT SECTION
    # ========================================================

    st.subheader("Step 1 — Enter Your Information")

    input_left, input_right = st.columns(2)

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    with input_left:

        st.markdown("#### Personal Information")

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
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

        family_history = st.selectbox(
            "Family history of overweight?",
            ["no", "yes"]
        )

        favc = st.selectbox(
            "Do you often eat high-calorie food?",
            ["no", "yes"]
        )

        smoke = st.selectbox(
            "Do you smoke?",
            ["no", "yes"]
        )

        scc = st.selectbox(
            "Do you monitor your calorie intake?",
            ["no", "yes"]
        )

    # --------------------------------------------------------
    # EATING + LIFESTYLE
    # --------------------------------------------------------

    with input_right:

        st.markdown("#### Eating Habits")

        fcvc = st.select_slider(
            "Vegetable consumption",
            options=[1.0, 2.0, 3.0],
            value=2.0,
            format_func=lambda x: {
                1.0: "Low",
                2.0: "Medium",
                3.0: "High"
            }[x]
        )

        ncp = st.select_slider(
            "Main meals per day",
            options=[1.0, 2.0, 3.0, 4.0],
            value=3.0,
            format_func=lambda x: (
                f"{int(x)} meal"
                if x == 1
                else f"{int(x)} meals"
            )
        )

        caec = st.selectbox(
            "Eating between meals",
            ["no", "Sometimes", "Frequently", "Always"],
            format_func=lambda x: "Never" if x == "no" else x
        )

        ch2o = st.select_slider(
            "Water intake",
            options=[1.0, 2.0, 3.0],
            value=2.0,
            format_func=lambda x: {
                1.0: "Low",
                2.0: "Medium",
                3.0: "High"
            }[x]
        )

        calc = st.selectbox(
            "Alcohol consumption",
            ["no", "Sometimes", "Frequently", "Always"],
            format_func=lambda x: "Never" if x == "no" else x
        )

        st.markdown("#### Lifestyle")

        faf = st.select_slider(
            "Physical activity",
            options=[0.0, 1.0, 2.0, 3.0],
            value=1.0,
            format_func=lambda x: {
                0.0: "Very Low",
                1.0: "Low",
                2.0: "Moderate",
                3.0: "High"
            }[x]
        )

        tue = st.select_slider(
            "Technology usage",
            options=[0.0, 1.0, 2.0],
            value=1.0,
            format_func=lambda x: {
                0.0: "Low",
                1.0: "Medium",
                2.0: "High"
            }[x]
        )

        mtrans = st.selectbox(
            "Main transportation",
            [
                "Automobile",
                "Bike",
                "Motorbike",
                "Public_Transportation",
                "Walking"
            ],
            format_func=lambda x: x.replace("_", " ")
        )

    # ========================================================
    # PREPARE INPUT FOR MODEL
    # ========================================================

    bmi = weight / (height ** 2)

    input_encoded = pd.DataFrame(
        0.0,
        index=[0],
        columns=feature_columns
    )

    # Numerical inputs
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

    for feature, value in categorical_values.items():
        dummy_column = f"{feature}_{value}"

        if dummy_column in input_encoded.columns:
            input_encoded.loc[0, dummy_column] = 1.0

    # ========================================================
    # AUTOMATIC PREDICTION
    # ========================================================

    input_scaled = scaler.transform(
        input_encoded
    )

    prediction = model.predict(
        input_scaled
    )

    predicted_label = label_encoder.inverse_transform(
        prediction.astype(int)
    )[0]

    friendly_label = predicted_label.replace(
        "_",
        " "
    )

    # Probability / confidence
    probability_df = None
    confidence = None

    if hasattr(model, "predict_proba"):

        probabilities = model.predict_proba(
            input_scaled
        )[0]

        model_classes = np.asarray(
            model.classes_
        ).astype(int)

        class_labels = label_encoder.inverse_transform(
            model_classes
        )

        probability_df = pd.DataFrame(
            {
                "Obesity Level": [
                    label.replace("_", " ")
                    for label in class_labels
                ],
                "Probability (%)": (
                    probabilities * 100
                ).round(2)
            }
        ).sort_values(
            "Probability (%)",
            ascending=False
        ).reset_index(drop=True)

        confidence = float(
            probability_df.iloc[0]["Probability (%)"]
        )

    # ========================================================
    # RESULT SECTION
    # ========================================================

    st.divider()
    st.subheader("Step 2 — Your Result")

    result_left, result_right = st.columns(
        [1.2, 1]
    )

    with result_left:

        st.success(
            f"## Predicted Obesity Level: {friendly_label}"
        )

        if confidence is not None:
            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )

            st.caption(
                "Model Confidence shows how strongly the model prefers "
                "this class compared with the other possible classes."
            )

        st.caption(
            "The result updates automatically whenever you change an input."
        )

    with result_right:

        st.metric(
            "BMI Reference",
            f"{bmi:.2f}"
        )

        st.caption(
            "BMI is displayed only as a reference. "
            "The model uses all entered features, not BMI alone."
        )

    # --------------------------------------------------------
    # TOP 3 PREDICTIONS
    # --------------------------------------------------------

    if probability_df is not None:

        st.markdown("#### Top 3 Possible Classes")

        top3 = probability_df.head(3)

        st.dataframe(
            top3,
            use_container_width=True,
            hide_index=True
        )

        top3_chart = px.bar(
            top3.sort_values(
                "Probability (%)",
                ascending=True
            ),
            x="Probability (%)",
            y="Obesity Level",
            orientation="h",
            text="Probability (%)",
            title="Top 3 Prediction Probabilities"
        )

        top3_chart.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        top3_chart.update_layout(
            xaxis_range=[0, 100],
            height=320
        )

        st.plotly_chart(
            top3_chart,
            use_container_width=True
        )

    # ========================================================
    # OPTIONAL DETAILS
    # ========================================================

    with st.expander(
        "See Lifestyle Profile Summary"
    ):

        st.write(
            """
            This section shows the lifestyle values you entered in a simple
            bar chart. Each bar is compared with the **maximum value available
            for that feature**.

            **Important:** A longer bar does not mean healthier. It only means
            a higher input value.
            """
        )

        # Human-friendly labels
        vegetable_level = {
            1.0: "Low",
            2.0: "Medium",
            3.0: "High"
        }[fcvc]

        water_level = {
            1.0: "Low",
            2.0: "Medium",
            3.0: "High"
        }[ch2o]

        activity_level = {
            0.0: "Very Low",
            1.0: "Low",
            2.0: "Moderate",
            3.0: "High"
        }[faf]

        technology_level = {
            0.0: "Low",
            1.0: "Medium",
            2.0: "High"
        }[tue]

        lifestyle_df = pd.DataFrame(
            {
                "Lifestyle Feature": [
                    "Vegetable Consumption",
                    "Main Meals",
                    "Water Intake",
                    "Physical Activity",
                    "Technology Usage"
                ],
                "Relative Level (%)": [
                    (fcvc / 3.0) * 100,
                    (ncp / 4.0) * 100,
                    (ch2o / 3.0) * 100,
                    (faf / 3.0) * 100,
                    (tue / 2.0) * 100
                ],
                "Your Input": [
                    f"{vegetable_level} ({fcvc:.0f}/3)",
                    f"{int(ncp)} meals/day",
                    f"{water_level} ({ch2o:.0f}/3)",
                    f"{activity_level} ({faf:.0f}/3)",
                    f"{technology_level} ({tue:.0f}/2)"
                ]
            }
        )

        # Easy-to-read horizontal bar chart
        profile_fig = px.bar(
            lifestyle_df,
            x="Relative Level (%)",
            y="Lifestyle Feature",
            orientation="h",
            text="Your Input",
            title="Your Lifestyle Inputs"
        )

        profile_fig.update_traces(
            textposition="outside"
        )

        profile_fig.update_layout(
            xaxis=dict(
                title="Relative Input Level (%)",
                range=[0, 115]
            ),
            yaxis=dict(
                title=""
            ),
            height=390,
            margin=dict(
                l=20,
                r=130,
                t=60,
                b=40
            ),
            showlegend=False
        )

        st.plotly_chart(
            profile_fig,
            use_container_width=True
        )

        st.caption(
            "Example: 100% means the highest selectable value for that "
            "feature. It does not mean 100% healthy."
        )

        st.dataframe(
            lifestyle_df[
                [
                    "Lifestyle Feature",
                    "Your Input"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

    with st.expander(
        "How the Model Processes Your Input"
    ):

        st.write(
            """
            The prediction follows four simple steps:

            **1. User Input**  
            The system receives the information entered above.

            **2. Encoding**  
            Text answers such as gender and transport type are converted
            into numbers using the same one-hot encoding format used
            during model training.

            **3. Standardisation**  
            The saved StandardScaler transforms the input into the same
            scale used during training.

            **4. Prediction**  
            The trained Decision Tree model receives the processed input
            and returns the predicted obesity level.
            """
        )

        with st.expander(
            "Show Encoded Input Values"
        ):
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
