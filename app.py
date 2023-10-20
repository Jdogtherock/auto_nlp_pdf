import streamlit as st
import polars as pl
from pdf_preprocessing import *
from pdf_eda import *
from zero_shot import *
from summarize import *
from NER import *
import zipfile
import io
from dotenv import load_dotenv
load_dotenv()
import nltk

with st.sidebar:
    st.image(
        "https://images.datacamp.com/image/upload/v1664812812/Machine_Learning_Lifecycle_2ffa5897a7.png"
    )
    st.title("AutoLLM - By Jeffrey Gordon")
    st.info(
        "This application automates the Generative LLM pipeline, by performing the most common Large Language Model Tasks with a simple click. Enter a PDF file with text, and watch the magic happen! :triumph:"
    )
    st.info(
        "NOTE: when the next step is ready, the button will appear below. To start over, refresh the webpage."
    )

# persisting variables i want to keep track of
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "pdf" not in st.session_state:
    st.session_state.pdf = None
if "text" not in st.session_state:
    st.session_state.text = None

ALLOWED_TYPES = ["pdf"]


def do_eda():
    st.session_state.page = "eda"


def do_nlp():
    st.session_state.page = "nlp"


def do_QA():
    st.session_state.page = "QA"


if st.session_state.page == "upload":
    st.title("Upload Your Data")
    st.info(
        "PDF Expectations: The application extracts text from the PDF, then removes the stopwords. If the remaining word length is less than 5, then its considered invalid."
    )

    file = None
    file = st.file_uploader("Upload a PDF file", type=ALLOWED_TYPES)

    if file is not None:
        st.write("PDF file uploaded.")
        st.session_state.pdf = file
        st.session_state.text = extract_text_from_pdf(file)
        if validity_check(st.session_state.text) == False:
            st.info(
                "The uploaded PDF is INVALID! Upload a PDF which contains words other than: stopwords, whitespace characters, images. Refresh to restart."
            )
        else:
            st.write("Text extracted from PDF.")
            st.write("First 50 words from the PDF:")
            st.write(first_n_words(st.session_state.text, 50))
            eda = st.button(
                "Exploratory Data Analysis", on_click=do_eda, type="primary"
            )

elif st.session_state.page == "eda":
    @st.cache_data(persist="disk")
    def perform_eda(text):
        with st.spinner("Performing EDA..."):
            eda_dict = combined_eda(text)
            pos_buf = pos_chart(text)
            word_freq_buf = top_bar_chart(text)
        return eda_dict, pos_buf, word_freq_buf
    eda_dict, pos_buf, word_freq_buf = perform_eda(st.session_state.text)
    st.title("Exploratory Data Analysis Results")
    # Display EDA results
    for key, value in eda_dict.items():
        if isinstance(value, dict):
            st.subheader(key)
            for subkey, subvalue in value.items():
                st.write(f"{subkey}: {subvalue}")
        else:
            st.write(f"{key}: {value}")
    # Display images from BytesIO buffers
    st.title("Visualizations")
    st.subheader("Part-of-Speech Distribution")
    st.image(pos_buf, caption="Part-of-Speech Distribution", use_column_width=True)
    st.subheader("Word Frequency Bar Chart")
    st.image(word_freq_buf, caption="Top 10 Frequent Words", use_column_width=True)
    # Download EDA Results & Images
    # Convert EDA dictionary to string
    eda_text = "\n".join([f"{key}: {value}" for key, value in eda_dict.items()])

   # Create checkboxes for selecting data to download
    eda_checkbox = st.checkbox("Include EDA Results in download")
    pos_chart_checkbox = st.checkbox("Include POS Pie Chart in download")
    word_freq_chart_checkbox = st.checkbox("Include Word Frequency Bar Chart in download")

    # If any checkbox is selected, create a download button
    if eda_checkbox or pos_chart_checkbox or word_freq_chart_checkbox:
        # Create a BytesIO buffer to hold the zipped files
        zip_buffer = io.BytesIO()

        # Create a zipfile object
        with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
            if eda_checkbox:
                # Write the EDA text to the zip file
                zip_file.writestr("eda_results.txt", eda_text)
            if pos_chart_checkbox:
                # Convert the BytesIO buffer back to bytes for the POS pie chart
                pos_bytes = pos_buf.getvalue()
                # Write the POS pie chart to the zip file
                zip_file.writestr("pos_pie_chart.png", pos_bytes)
            if word_freq_chart_checkbox:
                # Convert the BytesIO buffer back to bytes for the word frequency bar chart
                word_freq_bytes = word_freq_buf.getvalue()
                # Write the word frequency bar chart to the zip file
                zip_file.writestr("word_freq_bar_chart.png", word_freq_bytes)
        # Provide a download button for the zipped files
        st.download_button(
            "Download Selected Data",
            data=zip_buffer.getvalue(),
            file_name="selected_data.zip",
            mime="application/zip"
        )
    nlp = st.button("Auto NLP", on_click=do_nlp, type="primary")

elif st.session_state.page == "nlp":
    st.title("Automated NLP Results")

    @st.cache_data(persist="disk")
    def do_zero_shot_classify(text):
        with st.spinner("Performing Zero-Shot Classification..."):
            pred_label = zero_shot_classify(st.session_state.text)
        st.subheader("Zero-Shot Classification")
        st.write(f"This text is predicted to be about {pred_label}.")
        return pred_label

    pred_label = do_zero_shot_classify(st.session_state.text)

    @st.cache_data(persist="disk")
    def do_summarize(text):
        with st.spinner("Performing Summarization..."):
            summary = summarize_text(text)
        st.subheader("Summary")
        st.write(summary)
        return summary

    summary = do_summarize(st.session_state.text)

    @st.cache_data(persist="disk")
    def do_ner(text):
        with st.spinner(f"Performing Named Entity Recognition..."):
            ner_results = ner(text)
            ner_results = {key: list(value) for key, value in ner_results.items()}
        st.subheader("Named Entities")
        markdown_str = "| Key | Value |\n| --- | ----- |\n" + "\n".join(
            [f"| {key} | {value} |" for key, value in ner_results.items()]
        )
        st.markdown(markdown_str)
        key_descriptions = """\n| Key   | Description |\n|-------|-------------|\n| PER | PERSON |\n| MIS | MISCELLANEOUS |\n| LOC | LOCATION |\n| ORG | ORGANIZATION |\n"""
        st.markdown(key_descriptions)
        return markdown_str + "\n" + key_descriptions

    ner_results = do_ner(st.session_state.text)

    # Create checkboxes for selecting data to download
    classification_checkbox = st.checkbox("Include Classification in download")
    summary_checkbox = st.checkbox("Include Summary in download")
    ner_checkbox = st.checkbox("Include NER in download")

    # If any checkbox is selected, create a download button
    if classification_checkbox or summary_checkbox or ner_checkbox:
        download_str = ""

        if classification_checkbox:
            download_str += (
                "Zero-Shot Classification:\n"
                + f"This text is predicted to be about {pred_label}."
                + "\n\n"
            )

        if summary_checkbox:
            download_str += "Summary:\n" + summary + "\n\n"

        if ner_checkbox:
            ner_str = ner_results
            download_str += ner_str

        download_button = st.download_button(
            label="Download Selected Data",
            data=download_str,
            file_name="selected_data.txt",
            mime="text/plain",
        )

    chat = st.button("Chat With Your PDF", on_click=do_QA, type="primary")

elif st.session_state.page == "QA":
    st.title("PDF Chatbot")
    st.subheader("Document QA with LLM and VectorDB")
    st.image("QA_Design.png")

