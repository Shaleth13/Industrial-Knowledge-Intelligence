import streamlit as st
import plotly.express as px
from src.dataset import CMAPSSData
from src.model import RULModel
from src.predictor import EnginePredictor
from src.maintenance import MaintenanceAdvisor
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
st.set_page_config(
    page_title="Industrial Knowledge Intelligence",
    page_icon="⚙️",
    layout="wide")
st.title("⚙️ Industrial Knowledge Intelligence")
st.caption("Predictive Maintenance using NASA C-MAPSS Dataset")

# ---------------- Load Dataset ---------------- #

@st.cache_data
def load_dataset():
    data = CMAPSSData(
        BASE_DIR / "data" / "train_FD001.txt",
        BASE_DIR / "data" / "test_FD001.txt",
        BASE_DIR / "data" / "RUL_FD001.txt")
    return data.load()
train, test, rul = load_dataset()

# ---------------- Sidebar ---------------- #

st.sidebar.header("Actions")
if st.sidebar.button("Train Model"):
    model = RULModel()
    model.train(train)
    model.save(BASE_DIR / "models" / "xgboost_model.pkl")
    st.sidebar.success("Model Trained Successfully")

# ---------------- Prediction ---------------- #

if st.sidebar.button("Generate Predictions"):
    predictor = EnginePredictor(BASE_DIR / "models" / "xgboost_model.pkl")
    prediction = predictor.predict(test)
    latest = predictor.latest_prediction(prediction)
    advisor = MaintenanceAdvisor()
    result = advisor.analyze(latest)
    st.subheader("Prediction Results")
    st.dataframe(result, use_container_width=True)
    st.divider()
    healthy = len(result[result["Status"]=="Healthy"])
    warning = len(result[result["Status"]=="Warning"])
    critical = len(result[result["Status"]=="Critical"])
    
    c1,c2,c3 = st.columns(3)
    c1.metric("Healthy Engines",healthy)
    c2.metric("Warning",warning)
    c3.metric("Critical",critical)

    st.divider()
    engine = st.selectbox("Select Engine", result["engine_id"])
    selected = result[result["engine_id"]==engine]
    st.subheader("Engine Summary")
    st.dataframe(selected,use_container_width=True)

    fig = px.bar(
        selected,
        x="engine_id",
        y="Predicted_RUL",
        title="Predicted Remaining Useful Life")

    st.plotly_chart(fig,use_container_width=True)

    st.download_button("Download Predictions", result.to_csv(index=False), "predictions.csv", "text/csv")
