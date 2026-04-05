import time
from io import BytesIO

import pandas as pd
import requests
import streamlit as st

st.set_page_config(
    page_title="Marketing Campaign Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="📊",
)

REQUIRED_COLUMNS = [
    "age", "balance", "day", "campaign", "pdays", "previous",
    "job", "marital", "education", "default", "housing",
    "loan", "contact", "month", "poutcome",
]

SAMPLE_DATA = pd.DataFrame(
    {
        "age": [30, 45, 22, 60],
        "balance": [1000, -200, 3500, 800],
        "day": [5, 12, 20, 8],
        "campaign": [1, 3, 2, 1],
        "pdays": [-1, 90, -1, 180],
        "previous": [0, 2, 0, 1],
        "job": ["admin.", "blue-collar", "student", "management"],
        "marital": ["married", "single", "single", "divorced"],
        "education": ["secondary", "primary", "tertiary", "tertiary"],
        "default": ["no", "no", "no", "no"],
        "housing": ["yes", "no", "yes", "no"],
        "loan": ["no", "yes", "no", "no"],
        "contact": ["cellular", "unknown", "telephone", "cellular"],
        "month": ["jan", "jun", "mar", "nov"],
        "poutcome": ["unknown", "failure", "unknown", "success"],
    }
)


def get_file_key(uploaded_file) -> tuple:
    return (uploaded_file.name, uploaded_file.size)


def load_df(file_bytes: bytes, filename: str) -> pd.DataFrame | None:
    try:
        bio = BytesIO(file_bytes)
        with st.spinner("Reading file..."):
            if filename.lower().endswith(".csv"):
                return pd.read_csv(bio)
            elif filename.lower().endswith((".xlsx", ".xls")):
                return pd.read_excel(bio)
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                return None
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def get_mime(filename: str) -> str:
    name = filename.lower()
    if name.endswith(".csv"):
        return "text/csv"
    if name.endswith(".xlsx"):
        return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    if name.endswith(".xls"):
        return "application/vnd.ms-excel"
    return "application/octet-stream"


def send_to_api(file_bytes: bytes, filename: str, api_url: str) -> dict | None:
    try:
        files = {"file": (filename, file_bytes, get_mime(filename))}
        resp = requests.post(api_url, files=files, timeout=60)
        if resp.headers.get("content-type", "").startswith("application/json"):
            data = resp.json()
        else:
            st.error(f"Unexpected response from API (status {resp.status_code}): {resp.text[:400]}")
            return None

        if resp.status_code == 200:
            return data
        detail = data.get("detail") if isinstance(data, dict) else None
        st.error(f"API Error {resp.status_code}: {detail or data}")
        return None

    except requests.exceptions.RequestException as e:
        st.error(f"Network error while calling API: {e}")
        return None
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        return None


def main():
    if "file_key" not in st.session_state:
        st.session_state.file_key = None
    if "file_bytes" not in st.session_state:
        st.session_state.file_bytes = None
    if "df" not in st.session_state:
        st.session_state.df = None
    if "results" not in st.session_state:
        st.session_state.results = None

    # Sidebar
    with st.sidebar:
        st.title("📊 Marketing Campaign Predictor")
        st.write("Upload customer data to predict term deposit subscription likelihood.")
        st.markdown("---")

        st.markdown("### Upload Data")
        uploaded = st.file_uploader(
            "Choose a CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Supported: .csv, .xlsx, .xls",
        )

        st.markdown("---")
        st.markdown("### Settings")
        api_url = st.text_input("Backend API URL", value="http://backend:8000/predict")
        threshold = st.slider(
            "Decision Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Probability cutoff for predicting 'Yes'. Lower = more positives, higher = fewer.",
        )

    # Handle file upload
    if uploaded is not None:
        current_key = get_file_key(uploaded)
        if current_key != st.session_state.file_key:
            st.session_state.file_key = current_key
            st.session_state.file_bytes = uploaded.getvalue()
            st.session_state.df = load_df(st.session_state.file_bytes, uploaded.name)
            st.session_state.results = None
    else:
        st.session_state.file_key = None
        st.session_state.file_bytes = None
        st.session_state.df = None
        st.session_state.results = None

    # Main content
    st.title("Marketing Campaign Prediction Model")
    st.write("Predict which customers are likely to subscribe to a term deposit.")
    st.markdown("---")

    if st.session_state.df is not None:
        assert uploaded is not None
        assert st.session_state.file_bytes is not None

        st.success("File uploaded successfully — ready for prediction.")

        st.markdown("### Dataset Overview")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Filename", uploaded.name)
        c2.metric("Rows", f"{len(st.session_state.df):,}")
        c3.metric("Columns", len(st.session_state.df.columns))
        c4.metric("Size", f"{len(st.session_state.file_bytes or b'') / 1024:.1f} KB")

        st.markdown("### Data Preview")
        with st.expander("View uploaded data", expanded=True):
            st.dataframe(st.session_state.df.head(100), use_container_width=True, height=300)

        if st.checkbox("Show Statistical Summary"):
            st.dataframe(st.session_state.df.describe(), use_container_width=True)

        st.markdown("### Predict")
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in st.session_state.df.columns]
        if missing_cols:
            st.error(f"Uploaded file is missing required columns: {', '.join(missing_cols)}")

        _, mid, _ = st.columns([1, 1, 1])
        with mid:
            if st.button("Generate Predictions", use_container_width=True, type="primary", disabled=bool(missing_cols)):
                with st.spinner("Sending data to model..."):
                    start = time.time()
                    data = send_to_api(
                        st.session_state.file_bytes, uploaded.name, api_url
                    )
                    if data and isinstance(data, dict) and data.get("status") == "success":
                        d = data.get("data", {})
                        preds = d.get("predictions", [])
                        probas = d.get("probabilities", [])
                        proc = d.get("processing_time_seconds", None)
                        if len(preds) != len(st.session_state.df):
                            st.warning(
                                f"Prediction count ({len(preds)}) does not match rows "
                                f"({len(st.session_state.df)}). Truncating to min length."
                            )
                        n = min(len(preds), len(st.session_state.df))
                        st.session_state.results = {
                            "predictions": preds[:n],
                            "probabilities": probas[:n],
                            "processing_time": proc if proc is not None else round(time.time() - start, 3),
                            "num_predictions": d.get("num_predictions", n),
                        }
                    else:
                        st.session_state.results = None

        if st.session_state.results:
            r = st.session_state.results
            st.markdown("### Results")

            n = len(r["predictions"])
            probas = r.get("probabilities", [])
            out_df = st.session_state.df.head(n).copy()

            if probas:
                out_df["Probability"] = [round(p, 4) for p in probas]
                out_df["Prediction"] = ["Yes" if p >= threshold else "No" for p in probas]
            else:
                out_df["Prediction"] = ["Yes" if p == 1 else "No" for p in r["predictions"]]

            subscription_rate = out_df["Prediction"].eq("Yes").mean() * 100
            c1, c2, c3 = st.columns(3)
            c1.metric("Subscription Rate", f"{subscription_rate:.1f}%")
            c2.metric("Processing Time", f"{r.get('processing_time', 0):.3f}s")
            c3.metric("Predictions Made", r.get("num_predictions", len(r["predictions"])))

            st.dataframe(out_df, use_container_width=True, height=400)

            _, mid, _ = st.columns([1, 1, 1])
            with mid:
                st.download_button(
                    label="Download Predictions (CSV)",
                    data=out_df.to_csv(index=False),
                    file_name=f"predictions_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    else:
        st.markdown("### Get Started")
        st.info("Upload a CSV or Excel file using the sidebar to generate predictions.")

        st.markdown("### Expected Data Format")
        st.write("Your file should contain the following columns:")
        st.dataframe(SAMPLE_DATA, use_container_width=True)
        st.caption(
            "Each row represents one customer. "
            "The model predicts whether they will subscribe to a term deposit (1 = yes, 0 = no)."
        )


if __name__ == "__main__":
    main()
