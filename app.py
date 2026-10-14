
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Medical No-Show Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Health Care Medical Appointment No-Show Prediction")
st.write(
    "A machine learning application to predict whether a patient "
    "is likely to miss a scheduled medical appointment."
)

# --------------------------------------------------
# 1. DATA PREPROCESSING
# --------------------------------------------------

def preprocess_data(data):
    df = data.copy()
    df.columns = df.columns.str.strip()

    # Standardize common dataset column names
    df = df.rename(columns={
        "Hipertension": "Hypertension",
        "Handcap": "Handicap",
        "No-show": "No_show"
    })

    required = [
        "Gender", "ScheduledDay", "AppointmentDay", "Age",
        "Neighbourhood", "Scholarship", "Hypertension",
        "Diabetes", "Alcoholism", "Handicap",
        "SMS_received", "No_show"
    ]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    # Convert appointment dates
    df["ScheduledDay"] = pd.to_datetime(
        df["ScheduledDay"], errors="coerce", utc=True
    )
    df["AppointmentDay"] = pd.to_datetime(
        df["AppointmentDay"], errors="coerce", utc=True
    )

    # Feature engineering
    df["Scheduled_Hour"] = df["ScheduledDay"].dt.hour
    df["Appointment_DayOfWeek"] = (
        df["AppointmentDay"].dt.dayofweek
    )
    df["Appointment_Month"] = df["AppointmentDay"].dt.month

    df["Waiting_Days"] = (
        df["AppointmentDay"] - df["ScheduledDay"]
    ).dt.total_seconds() / 86400

    # Convert target: No = attended, Yes = missed
    df["No_show"] = (
        df["No_show"].astype(str).str.strip().str.lower()
        .map({"no": 0, "yes": 1, "0": 0, "1": 1})
    )

    # Remove invalid records
    df = df.dropna(subset=[
        "No_show", "ScheduledDay", "AppointmentDay",
        "Age", "Waiting_Days"
    ])

    df = df[(df["Age"] >= 0) & (df["Waiting_Days"] >= 0)]

    features = [
        "Gender", "Age", "Neighbourhood", "Scholarship",
        "Hypertension", "Diabetes", "Alcoholism", "Handicap",
        "SMS_received", "Scheduled_Hour",
        "Appointment_DayOfWeek", "Appointment_Month",
        "Waiting_Days"
    ]

    X = df[features].copy()
    y = df["No_show"].astype(int)

    # Standardize data types
    categorical_features = ["Gender", "Neighbourhood"]

    for col in categorical_features:
        X[col] = X[col].fillna("Unknown").astype(str)

    numeric_features = [
        col for col in features if col not in categorical_features
    ]

    for col in numeric_features:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    return X, y, features, categorical_features, numeric_features


# --------------------------------------------------
# 2. MODEL TRAINING
# --------------------------------------------------

def create_pipeline(model, categorical_features, numeric_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])


# --------------------------------------------------
# 3. DATASET UPLOAD
# --------------------------------------------------

st.sidebar.header("📂 Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your medical appointment CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Upload KaggleV2-May-2016.csv using the sidebar to begin."
    )
    st.write("Expected target column: `No-show`")
    st.write(
        "The application automatically creates appointment "
        "time, weekday, month, and waiting-day features."
    )
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
    X, y, features, cat_features, num_features = preprocess_data(
        raw_df
    )
except Exception as error:
    st.error(f"Could not process the dataset: {error}")
    st.stop()

if y.nunique() < 2:
    st.error("The dataset must contain both attended and missed appointments.")
    st.stop()

st.success("Dataset loaded and preprocessed successfully.")

# --------------------------------------------------
# 4. DATASET OVERVIEW
# --------------------------------------------------

st.header("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Valid Appointments", f"{len(X):,}")
c2.metric("Patient Features", len(features))
c3.metric("Attended", f"{(y == 0).sum():,}")
c4.metric("Missed", f"{(y == 1).sum():,}")

with st.expander("Preview processed dataset"):
    st.dataframe(X.head(10), use_container_width=True)

with st.expander("Appointment attendance distribution"):
    counts = y.map({
        0: "Attended",
        1: "Missed"
    }).value_counts()

    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values)
    ax.set_title("Appointment Attendance Distribution")
    ax.set_ylabel("Number of Appointments")
    st.pyplot(fig)
    plt.close(fig)

# --------------------------------------------------
# 5. TRAIN MODELS
# --------------------------------------------------

st.header("🤖 Model Training")

if st.button("Train Random Forest and XGBoost", type="primary"):
    with st.spinner("Training models. Please wait..."):
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
            stratify=y
        )

        models = {
            "Random Forest": RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            ),
            "XGBoost": XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1
            )
        }

        results = []
        trained_models = {}

        progress = st.progress(0)

        for i, (name, model) in enumerate(models.items()):
            pipeline = create_pipeline(
                model, cat_features, num_features
            )

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(
                    y_test, predictions, zero_division=0
                ),
                "Recall": recall_score(
                    y_test, predictions, zero_division=0
                ),
                "F1-score": f1_score(
                    y_test, predictions, zero_division=0
                )
            })

            trained_models[name] = pipeline
            progress.progress((i + 1) / len(models))

        results_df = pd.DataFrame(results)
        best_name = results_df.sort_values(
            "F1-score", ascending=False
        ).iloc[0]["Model"]

        best_model = trained_models[best_name]
        best_predictions = best_model.predict(X_test)

        # Store the trained model and test results for later UI use
        st.session_state["trained_models"] = trained_models
        st.session_state["best_name"] = best_name
        st.session_state["results_df"] = results_df
        st.session_state["best_model"] = best_model
        st.session_state["features"] = features
        st.session_state["cat_features"] = cat_features
        st.session_state["num_features"] = num_features
        st.session_state["X_test"] = X_test
        st.session_state["y_test"] = y_test
        st.session_state["best_predictions"] = best_predictions

    st.success("Both models have been trained successfully.")

# --------------------------------------------------
# 6. MODEL EVALUATION
# --------------------------------------------------

if "results_df" in st.session_state:
    st.subheader("Model Performance Comparison")

    results_df = st.session_state["results_df"]
    st.dataframe(
        results_df.style.format({
            "Accuracy": "{:.3f}",
            "Precision": "{:.3f}",
            "Recall": "{:.3f}",
            "F1-score": "{:.3f}"
        }),
        use_container_width=True
    )

    best_name = st.session_state["best_name"]
    st.success(f"Selected model based on F1-score: {best_name}")

    st.subheader("Confusion Matrix")

    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        st.session_state["y_test"],
        st.session_state["best_predictions"],
        display_labels=["Attended", "Missed"],
        cmap="Blues",
        ax=ax
    )
    ax.set_title(f"{best_name} Confusion Matrix")
    st.pyplot(fig)
    plt.close(fig)

    # Feature importance, when supported by the selected model
    st.subheader("Feature Importance")

    fitted_pipeline = st.session_state["best_model"]
    fitted_preprocessor = fitted_pipeline.named_steps["preprocessor"]
    fitted_model = fitted_pipeline.named_steps["model"]

    if hasattr(fitted_model, "feature_importances_"):
        try:
            transformed_names = fitted_preprocessor.get_feature_names_out()
            importance = fitted_model.feature_importances_

            importance_df = pd.DataFrame({
                "Feature": transformed_names,
                "Importance": importance
            }).sort_values("Importance", ascending=False).head(15)

            fig, ax = plt.subplots()
            ax.barh(
                importance_df["Feature"][::-1],
                importance_df["Importance"][::-1]
            )
            ax.set_title("Top 15 Important Features")
            ax.set_xlabel("Importance")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as error:
            st.warning(f"Could not display feature importance: {error}")

# --------------------------------------------------
# 7. PATIENT PREDICTION
# --------------------------------------------------

st.header("🩺 Predict Appointment Attendance")

if "best_model" not in st.session_state:
    st.info("Train both models above before making a prediction.")
    st.stop()

st.write("Enter the patient's details below.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["F", "M"])
        age = st.number_input(
            "Age", min_value=0, max_value=120, value=35
        )
        neighbourhood = st.selectbox(
            "Neighbourhood",
            sorted(X["Neighbourhood"].unique().tolist())
        )
        scholarship = st.selectbox("Scholarship", [0, 1])

    with col2:
        hypertension = st.selectbox("Hypertension", [0, 1])
        diabetes = st.selectbox("Diabetes", [0, 1])
        alcoholism = st.selectbox("Alcoholism", [0, 1])
        handicap = st.selectbox("Handicap", [0, 1, 2, 3, 4])

    with col3:
        sms_received = st.selectbox("SMS Received", [0, 1])
        scheduled_hour = st.slider(
            "Scheduled Hour", 0, 23, 10
        )
        appointment_day = st.selectbox(
            "Appointment Weekday",
            options=list(range(7)),
            format_func=lambda x: [
                "Monday", "Tuesday", "Wednesday", "Thursday",
                "Friday", "Saturday", "Sunday"
            ][x]
        )
        appointment_month = st.selectbox(
            "Appointment Month", list(range(1, 13)), index=5
        )
        waiting_days = st.number_input(
            "Waiting Days", min_value=0.0, max_value=365.0,
            value=7.0, step=1.0
        )

    submitted = st.form_submit_button(
        "Predict Appointment Outcome",
        type="primary"
    )

if submitted:
    patient = pd.DataFrame([{
        "Gender": gender,
        "Age": age,
        "Neighbourhood": neighbourhood,
        "Scholarship": scholarship,
        "Hypertension": hypertension,
        "Diabetes": diabetes,
        "Alcoholism": alcoholism,
        "Handicap": handicap,
        "SMS_received": sms_received,
        "Scheduled_Hour": scheduled_hour,
        "Appointment_DayOfWeek": appointment_day,
        "Appointment_Month": appointment_month,
        "Waiting_Days": waiting_days
    }])

    # Keep the same feature order as model training
    patient = patient[st.session_state["features"]]

    model = st.session_state["best_model"]
    prediction = int(model.predict(patient)[0])

    probabilities = model.predict_proba(patient)[0]
    classes = list(model.named_steps["model"].classes_)
    missed_probability = float(probabilities[classes.index(1)])

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning("⚠️ The model predicts that the patient may MISS the appointment.")
    else:
        st.success("✅ The model predicts that the patient may ATTEND the appointment.")

    st.metric(
        "Estimated No-Show Probability",
        f"{missed_probability:.1%}"
    )

    st.caption(
        "This is a machine-learning estimate, not a guarantee. "
        "Use it as decision support, not as a substitute for clinical judgment."
    )
