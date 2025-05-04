
import streamlit as st
import pandas as pd

st.title("Sample Streamlit App")
df = pd.DataFrame({
    "A": [1, 2, 3],
    "B": [4, 5, 6]
})
st.write(df)
