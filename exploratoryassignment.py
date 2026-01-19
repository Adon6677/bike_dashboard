import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bike Rental Analysis", layout="wide")

# =========================
# Load data ONCE
# =========================
df = pd.read_csv("train.csv")
df["datetime"] = pd.to_datetime(df["datetime"])

# Feature engineering
df["year"] = df["datetime"].dt.year
df["month"] = df["datetime"].dt.month
df["hour"] = df["datetime"].dt.hour
df["weekday"] = df["datetime"].dt.dayofweek

season_map = {1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"}
df["season"] = df["season"].map(season_map)

# =========================
# Sidebar
# =========================
st.sidebar.header("Filters")

year = st.sidebar.selectbox("Year", sorted(df["year"].unique()))
working_only = st.sidebar.checkbox("Working days only")

filtered_df = df[df["year"] == year]
if working_only:
    filtered_df = filtered_df[filtered_df["workingday"] == 1]

# =========================
# App Title
# =========================
st.title("🚲 Bike Rental Exploratory Data Analysis")

st.write("### Data Preview")
st.dataframe(filtered_df.head())

# =========================
# Mean Rentals by Hour
# =========================
st.write("### Mean Hourly Rentals")

hourly = filtered_df.groupby("hour")["count"].mean()
fig, ax = plt.subplots()
hourly.plot(ax=ax)
ax.set_xlabel("Hour")
ax.set_ylabel("Mean Rentals")
st.pyplot(fig)

# =========================
# Working vs Non-working Days
# =========================
st.write("### Working vs Non-working Days")

mean_working = df.groupby("workingday")["count"].mean()
fig, ax = plt.subplots()
mean_working.plot(kind="bar", ax=ax)
ax.set_xlabel("Working Day (0 = No, 1 = Yes)")
ax.set_ylabel("Mean Rentals")
st.pyplot(fig)

# =========================
# Monthly Trend
# =========================
st.write("### Mean Monthly Rentals")

monthly_mean = df.groupby("month")["count"].mean()
fig, ax = plt.subplots()
monthly_mean.plot(marker="o", ax=ax)
ax.set_xticks(range(1, 13))
ax.set_xlabel("Month")
ax.set_ylabel("Mean Rentals")
st.pyplot(fig)

# =========================
# Correlation Heatmap
# =========================
st.write("### Correlation Matrix")

num_cols = df.select_dtypes(include=np.number).columns
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
st.pyplot(fig)
