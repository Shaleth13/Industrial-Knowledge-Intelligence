import streamlit as st
from src.dataset import CMAPSSData
from src.model import RULModel
from src.predictor import EnginePredictor
from src.maintenance import MaintenanceAdvisor


st.set_page_config(
    page_title="Industrial Knowledge Intelligence",
    page_icon="⚙️",
    layout="wide")

st.title("⚙️ Industrial Knowledge Intelligence")
st.write("Predictive Maintenance using NASA C-MAPSS Dataset")

@st.cache_data
def load_dataset():
    data = CMAPSSData(
        "data/train_FD001.txt",
        "data/test_FD001.txt",
        "data/RUL_FD001.txt")
    return data.load()

train, test, rul = load_dataset()

if st.sidebar.button("Train Model"):
    model = RULModel()
    model.train(train)
    model.save()
    st.sidebar.success("Model Trained Successfully")

if st.sidebar.button("Predict"):
    predictor = EnginePredictor()
    prediction = predictor.predict(test)
    latest = predictor.latest_prediction(prediction)
    advisor = MaintenanceAdvisor()
    result = advisor.analyze(latest)
    st.dataframe(result)
    st.subheader("Engine Health Summary")

    healthy = len(result[result["Status"] == "Healthy"])
    warning = len(result[result["Status"] == "Warning"])
    critical = len(result[result["Status"] == "Critical"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Healthy", healthy)
    c2.metric("Warning", warning)
    c3.metric("Critical", critical)

    engine = st.selectbox("Select Engine", result["engine_id"])
    selected = result[result["engine_id"] == engine]
    st.subheader("Prediction")
    st.write(selected)
