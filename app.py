import streamlit as st
from load_models import get_real_summaries

st.set_page_config(page_title="LLM Summarizer Comparator", layout="wide")

st.title("LLM Summarizer Comparator")
st.markdown("Compare summarization outputs from GPT-3.5 (placeholder), BART, and T5 on the same input text.")

user_input = st.text_area("Enter text to summarize", height=200)

if st.button("Generate Summaries"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        summaries = get_real_summaries(user_input)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("GPT-3.5 (placeholder)")
            st.write(summaries['gpt'])

        with col2:
            st.subheader("BART")
            st.write(summaries['bart'])
            st.caption(f"Length: {summaries['bart_metrics']['length']} words | ROUGE-L: {summaries['bart_metrics']['rougeL']}")

        with col3:
            st.subheader("T5")
            st.write(summaries['t5'])
            st.caption(f"Length: {summaries['t5_metrics']['length']} words | ROUGE-L: {summaries['t5_metrics']['rougeL']}")
