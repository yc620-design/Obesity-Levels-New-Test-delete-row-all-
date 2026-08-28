import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        "Data Understanding",
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

    st.warning(
        "This prototype is for educational decision support and is not "
        "a medical diagnostic system."
    )


# ============================================================
# DATA UNDERSTANDING
# ============================================================

elif page == "Data Understanding":

    st.title("Data Understanding")

    if raw_df is None:
        st.error(
            "Add `ObesityDataSet_raw_and_data_sinthetic.csv` "
            "to the same folder as app.py."
        )
        st.stop()

    st.caption(
        f"Loaded file: {dataset_name}"
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Rows", f"{raw_df.shape[0]:,}")
    c2.metric("Columns", raw_df.shape[1])
    c3.metric(
        "Numerical Columns",
        raw_df.select_dtypes(
            include=np.number
        ).shape[1]
    )
    c4.metric(
        "Categorical Columns",
        raw_df.select_dtypes(
            exclude=np.number
        ).shape[1]
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        raw_df.head(10),
        use_container_width=True
    )

    st.subheader("Summary Statistics")

    st.dataframe(
        raw_df.describe().T.round(3),
        use_container_width=True
    )

    st.subheader("Target Class Distribution")

    class_counts = (
        raw_df["NObeyesdad"]
        .value_counts()
        .rename_axis("Obesity Level")
        .reset_index(name="Records")
    )

    class_counts["Percentage"] = (
        class_counts["Records"]
        / class_counts["Records"].sum()
        * 100
    ).round(2)

    st.dataframe(
        class_counts,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Data Quality Snapshot")

    q1, q2, q3 = st.columns(3)

    q1.metric(
        "Missing Values",
        int(raw_df.isnull().sum().sum())
    )

    q2.metric(
        "Exact Duplicates",
        int(raw_df.duplicated().sum())
    )

    q3.metric(
        "Target Classes",
        raw_df["NObeyesdad"].nunique()
    )


# ============================================================
# DATA ANALYSIS — 5 GRAPHS
# ============================================================

elif page == "Data Analysis":

    st.title("Data Analysis")

    if raw_df is None:
        st.error(
            "Dataset CSV is required to display the analysis graphs."
        )
        st.stop()

    st.write(
        """
        Five visualisations are provided to explore class distribution,
        eating habits, lifestyle behaviour and relationships among
        physical and numerical variables.
        """
    )

    # --------------------------------------------------------
    # GRAPH 1 — CLASS DISTRIBUTION
    # --------------------------------------------------------

    st.subheader("1. Distribution of Obesity Levels")

    class_counts = (
        raw_df["NObeyesdad"]
        .value_counts()
        .reindex(CLASS_ORDER)
        .dropna()
    )

    fig1, ax1 = plt.subplots(figsize=(11, 5.5))

    bars = ax1.bar(
        class_counts.index,
        class_counts.values
    )

    ax1.set_title(
        "Distribution of Obesity Levels"
    )

    ax1.set_xlabel(
        "Obesity Level"
    )

    ax1.set_ylabel(
        "Number of Records"
    )

    ax1.tick_params(
        axis="x",
        rotation=35
    )

    for bar in bars:
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    fig1.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.caption(
        "The classes are relatively close in size in the original dataset, "
        "although they are not perfectly balanced."
    )

    # --------------------------------------------------------
    # GRAPH 2 — CAEC BY OBESITY LEVEL
    # --------------------------------------------------------

    st.subheader(
        "2. Eating Between Meals (CAEC) by Obesity Level"
    )

    caec_table = pd.crosstab(
        raw_df["CAEC"],
        raw_df["NObeyesdad"]
    )

    caec_order = [
        item for item in [
            "no",
            "Sometimes",
            "Frequently",
            "Always"
        ]
        if item in caec_table.index
    ]

    caec_table = caec_table.reindex(
        caec_order
    )

    caec_table = caec_table.reindex(
        columns=[
            c for c in CLASS_ORDER
            if c in caec_table.columns
        ]
    )

    fig2, ax2 = plt.subplots(figsize=(12, 6))

    caec_table.plot(
        kind="bar",
        ax=ax2
    )

    ax2.set_title(
        "Eating Between Meals (CAEC) Broken Down by Obesity Level"
    )

    ax2.set_xlabel(
        "Frequency of Snacking Between Meals"
    )

    ax2.set_ylabel(
        "Number of Individuals"
    )

    ax2.tick_params(
        axis="x",
        rotation=0
    )

    ax2.legend(
        title="Obesity Level",
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)

    st.caption(
        "This chart compares snacking-frequency categories across all "
        "seven obesity levels."
    )

    # --------------------------------------------------------
    # GRAPH 3 — FAF VS TUE
    # --------------------------------------------------------

    st.subheader(
        "3. Average Physical Activity (FAF) vs Technology Time (TUE)"
    )

    habit_means = (
        raw_df
        .groupby("NObeyesdad")[["FAF", "TUE"]]
        .mean()
        .reindex(CLASS_ORDER)
        .dropna()
    )

    fig3, ax3 = plt.subplots(figsize=(12, 6))

    habit_means.plot(
        kind="bar",
        ax=ax3
    )

    ax3.set_title(
        "Average Physical Activity and Technology Use by Obesity Level"
    )

    ax3.set_xlabel(
        "Obesity Level"
    )

    ax3.set_ylabel(
        "Average Score"
    )

    ax3.tick_params(
        axis="x",
        rotation=35
    )

    ax3.legend(
        title="Lifestyle Feature"
    )

    fig3.tight_layout()
    st.pyplot(fig3)
    plt.close(fig3)

    st.caption(
        "FAF represents physical-activity frequency and TUE represents "
        "technology-use time."
    )

    # --------------------------------------------------------
    # GRAPH 4 — HEIGHT VS WEIGHT
    # --------------------------------------------------------

    st.subheader("4. Height vs Weight")

    x = raw_df["Height"].to_numpy()
    y = raw_df["Weight"].to_numpy()

    slope, intercept = np.polyfit(
        x,
        y,
        1
    )

    x_line = np.linspace(
        x.min(),
        x.max(),
        100
    )

    y_line = (
        slope * x_line
        + intercept
    )

    fig4, ax4 = plt.subplots(figsize=(9, 6))

    ax4.scatter(
        x,
        y,
        alpha=0.45
    )

    ax4.plot(
        x_line,
        y_line
    )

    ax4.set_title(
        "Scatterplot: Height vs Weight"
    )

    ax4.set_xlabel(
        "Height (m)"
    )

    ax4.set_ylabel(
        "Weight (kg)"
    )

    fig4.tight_layout()
    st.pyplot(fig4)
    plt.close(fig4)

    correlation = raw_df[
        ["Height", "Weight"]
    ].corr().iloc[0, 1]

    st.caption(
        f"Pearson correlation between Height and Weight: "
        f"{correlation:.3f}."
    )

    # --------------------------------------------------------
    # GRAPH 5 — CORRELATION HEATMAP
    # --------------------------------------------------------

    st.subheader("5. Correlation Heatmap")

    numeric_df = raw_df.select_dtypes(
        include=np.number
    )

    corr = numeric_df.corr()

    fig5, ax5 = plt.subplots(figsize=(10, 8))

    heat = ax5.imshow(
        corr.to_numpy(),
        aspect="auto"
    )

    ax5.set_xticks(
        np.arange(len(corr.columns))
    )

    ax5.set_yticks(
        np.arange(len(corr.columns))
    )

    ax5.set_xticklabels(
        corr.columns,
        rotation=45,
        ha="right"
    )

    ax5.set_yticklabels(
        corr.columns
    )

    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax5.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7
            )

    fig5.colorbar(
        heat,
        ax=ax5,
        fraction=0.046,
        pad=0.04
    )

    ax5.set_title(
        "Correlation Heatmap for Numerical Features"
    )

    fig5.tight_layout()
    st.pyplot(fig5)
    plt.close(fig5)

    st.caption(
        "The heatmap summarises pairwise linear relationships among "
        "the numerical variables."
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

    final_train = int(
        cleaned_df.shape[0] * 0.8
    )

    st.info(
        "The actual notebook split contains 701 training records and "
        "176 testing records, with 23 encoded input features."
    )

    st.warning(
        "Limitation: the interpolation filtering rule removes a large "
        "proportion of the duplicate-cleaned dataset. This should be "
        "reported transparently because it can affect class representation "
        "and generalisability."
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

    st.title("🔍 Obesity Level Prediction")

    if (
        model is None
        or scaler is None
        or label_encoder is None
        or feature_columns is None
    ):
        st.error(
            "Prediction files are missing. Add the following files "
            "to the same folder as app.py:"
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
        **How to use:** Fill in or change the information below.
        The prediction updates automatically using the trained
        machine-learning model.
        """
    )

    # --------------------------------------------------------
    # STEP 1 — PERSONAL INFORMATION
    # --------------------------------------------------------

    st.subheader("Step 1 — Personal Information")

    col1, col2 = st.columns(2)

    with col1:
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

    with col2:
        family_history = st.selectbox(
            "Family history of overweight?",
            ["no", "yes"],
            help="Select yes if close family members have a history of overweight."
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

    bmi = weight / (height ** 2)

    bmi_col1, bmi_col2 = st.columns([1, 2])

    with bmi_col1:
        st.metric(
            "BMI Reference",
            f"{bmi:.2f}"
        )

    with bmi_col2:
        st.caption(
            "BMI is shown only as a reference. The model prediction uses "
            "all the information you enter, not BMI alone."
        )

    # --------------------------------------------------------
    # STEP 2 — EATING HABITS
    # --------------------------------------------------------

    st.subheader("Step 2 — Eating Habits")

    col3, col4 = st.columns(2)

    with col3:
        fcvc = st.select_slider(
            "How often do you eat vegetables?",
            options=[1.0, 2.0, 3.0],
            value=2.0,
            format_func=lambda x: {
                1.0: "Low",
                2.0: "Medium",
                3.0: "High"
            }[x]
        )

        ncp = st.select_slider(
            "How many main meals do you usually eat per day?",
            options=[1.0, 2.0, 3.0, 4.0],
            value=3.0,
            format_func=lambda x: f"{int(x)} meal" if x == 1 else f"{int(x)} meals"
        )

        caec = st.selectbox(
            "How often do you eat between meals?",
            [
                "no",
                "Sometimes",
                "Frequently",
                "Always"
            ],
            format_func=lambda x: {
                "no": "Never",
                "Sometimes": "Sometimes",
                "Frequently": "Frequently",
                "Always": "Always"
            }[x]
        )

    with col4:
        ch2o = st.select_slider(
            "How much water do you usually drink?",
            options=[1.0, 2.0, 3.0],
            value=2.0,
            format_func=lambda x: {
                1.0: "Low",
                2.0: "Medium",
                3.0: "High"
            }[x]
        )

        calc = st.selectbox(
            "How often do you consume alcohol?",
            [
                "no",
                "Sometimes",
                "Frequently",
                "Always"
            ],
            format_func=lambda x: {
                "no": "Never",
                "Sometimes": "Sometimes",
                "Frequently": "Frequently",
                "Always": "Always"
            }[x]
        )

    # --------------------------------------------------------
    # STEP 3 — LIFESTYLE
    # --------------------------------------------------------

    st.subheader("Step 3 — Lifestyle")

    col5, col6 = st.columns(2)

    with col5:
        faf = st.select_slider(
            "How active are you physically?",
            options=[0.0, 1.0, 2.0, 3.0],
            value=1.0,
            format_func=lambda x: {
                0.0: "Very low",
                1.0: "Low",
                2.0: "Moderate",
                3.0: "High"
            }[x]
        )

        tue = st.select_slider(
            "How much time do you spend using technology?",
            options=[0.0, 1.0, 2.0],
            value=1.0,
            format_func=lambda x: {
                0.0: "Low",
                1.0: "Medium",
                2.0: "High"
            }[x]
        )

    with col6:
        mtrans = st.selectbox(
            "Main mode of transportation",
            [
                "Automobile",
                "Bike",
                "Motorbike",
                "Public_Transportation",
                "Walking"
            ],
            format_func=lambda x: x.replace("_", " ")
        )

    # --------------------------------------------------------
    # BUILD MODEL INPUT
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # AUTOMATIC PREDICTION
    # --------------------------------------------------------

    st.divider()

    st.info(
        "⚡ **Automatic Prediction:** The result updates automatically "
        "whenever you change any information above."
    )

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

    st.subheader("Your Prediction Result")

    st.success(
        f"### Predicted Obesity Level: {friendly_label}"
    )

    # ----------------------------------------------------
    # SIMPLE RESULT SUMMARY
    # ----------------------------------------------------

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "BMI Reference",
        f"{bmi:.2f}"
    )

    r2.metric(
        "Physical Activity",
        {
            0.0: "Very Low",
            1.0: "Low",
            2.0: "Moderate",
            3.0: "High"
        }[faf]
    )

    r3.metric(
        "Water Intake",
        {
            1.0: "Low",
            2.0: "Medium",
            3.0: "High"
        }[ch2o]
    )

    # ----------------------------------------------------
    # PROBABILITIES
    # ----------------------------------------------------

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

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )

        st.caption(
            "Model confidence means how strongly the model prefers "
            "this class compared with the other classes."
        )

        st.subheader("Top 3 Possible Classes")

        top3 = probability_df.head(3).copy()

        st.dataframe(
            top3,
            use_container_width=True,
            hide_index=True
        )

        fig_prob = px.bar(
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

        fig_prob.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig_prob.update_layout(
            xaxis_range=[0, 100],
            height=330
        )

        st.plotly_chart(
            fig_prob,
            use_container_width=True
        )

    # ----------------------------------------------------
    # OPTIONAL EXTRA VISUAL
    # ----------------------------------------------------

    with st.expander(
        "See Lifestyle Profile Chart"
    ):

        labels = [
            "Vegetables",
            "Meals",
            "Water",
            "Activity",
            "Technology"
        ]

        values = [
            ((fcvc - 1.0) / 2.0) * 100,
            ((ncp - 1.0) / 3.0) * 100,
            ((ch2o - 1.0) / 2.0) * 100,
            (faf / 3.0) * 100,
            (tue / 2.0) * 100
        ]

        radar_fig = go.Figure()

        radar_fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                name="Current Profile"
            )
        )

        radar_fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )
            ),
            showlegend=False,
            height=350
        )

        st.plotly_chart(
            radar_fig,
            use_container_width=True
        )

        st.caption(
            "This chart only visualises the entered lifestyle values. "
            "It is not a health score."
        )

    with st.expander(
        "How the Model Processes Your Input"
    ):
        st.write(
            """
            The system converts the entered information into the same
            encoded feature format used during model training, applies
            the saved StandardScaler, and sends the processed values to
            the trained model.
            """
        )

        st.dataframe(
            input_encoded,
            use_container_width=True
        )

    st.warning(
        "This prediction is for educational purposes only and is not "
        "a medical diagnosis."
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "BMDS2003 Data Science — Obesity Level Classification Prototype"
)
