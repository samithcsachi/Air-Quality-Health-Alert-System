import streamlit as st
from streamlit_autorefresh import st_autorefresh
import random
import pandas as pd
import plotly.express as px

from SRC.Air_Quality_Health_Alert_System.components.alert_generation import AlertGenerator
from SRC.Air_Quality_Health_Alert_System.config.configuration import ConfigurationManager


config_manager = ConfigurationManager()
alert_config = config_manager.get_alert_generation_config()
alert_generator = AlertGenerator(alert_config)

st.set_page_config(page_title="Air Quality Alert Dashboard", layout="wide")

st.title("🌍 Air Quality Alert Dashboard")


predicted_aqi = st.slider("Select Predicted AQI:", min_value=0, max_value=500, value=120, step=5)
st.session_state["predicted_aqi"] = predicted_aqi


actual_aqi = predicted_aqi + random.randint(-30, 30)
st.session_state["actual_aqi"] = actual_aqi


alerts = alert_generator.get_alerts_for_dashboard(predicted_aqi)

with st.container():
    if alerts:
        for msg in alerts:
            st.warning(msg)
    else:
        st.success("AQI is within safe limits.")


st.subheader("💡 Health Recommendations")
if predicted_aqi <= 50:
    st.info("Air quality is good. Enjoy your day outdoors!")
elif predicted_aqi <= 100:
    st.info("Air quality is moderate. Sensitive groups should take care.")
elif predicted_aqi <= 200:
    st.warning("Unhealthy for sensitive groups. Consider limiting outdoor activities.")
else:
    st.error("Hazardous! Stay indoors and use an air purifier if possible.")


if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.history.append({
    "predicted": predicted_aqi,
    "actual": actual_aqi,
    "alert": alerts[0] if alerts else "Safe"
})


if len(st.session_state.history) > 10:
    st.session_state.history.pop(0)

history_df = pd.DataFrame(st.session_state.history)


st.subheader("📈 Predicted vs Actual AQI (Last 10 Updates)")
fig1 = px.line(
    history_df,
    y=["predicted", "actual"],
    markers=True,
    title="Predicted vs Actual AQI",
    line_shape="spline"
)
st.plotly_chart(fig1, use_container_width=True)


st.subheader("🚨 Alert History (Last 10 Updates)")
alert_df = history_df.copy()
alert_df["update"] = range(len(alert_df))
fig2 = px.scatter(
    alert_df,
    x="update",
    y="predicted",
    color="alert",
    title="Alert Levels Over Time",
    symbol="alert"
)
st.plotly_chart(fig2, use_container_width=True)

# Auto-refresh
st_autorefresh(interval=5000, limit=None, key="fizzbuzz_counter")
