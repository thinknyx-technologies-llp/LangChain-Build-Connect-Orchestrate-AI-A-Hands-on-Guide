import streamlit as st
import pandas as pd

from data_analyst import create_data_analyst, ask_agent

st.set_page_config(
  page_title="Data Analyst",
  layout="wide"  
)

st.title("ThinkSight: Your Personal AI Data Analyst")

st.write(
    "Upload any csv file and ask questions about you data  using natural language."
)

uploaded_file = st.file_uploader(
    "Upload your CSV file",
    type=["csv"]
)

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if uploaded_file is not None:

    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1"]

    df = None

    for encoding in encodings:
        try:
            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                encoding=encoding
            )

            break

        except UnicodeDecodeError:
            continue

    if df is None:
        st.error("Unable to read the csv file. Please check the file encoding")
        st.stop()


    st.success(
        f"Successfully loaded: {uploaded_file.name}"
    )

    st.subheader("Data Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric(
            "Missing Values", int(df.isnull().sum().sum())
        )

    st.subheader("Data Preview")

    st.dataframe(
    df.head(20),
    use_container_width=True 
    )

    with st.expander(
        "View Dataset schema"
    ):
        schema = pd.DataFrame(
            {
                "Columns": df.columns,
                "Data Type": df.dtypes.astype(str),
                "Missing Values": df.isnull().sum().values
            }
        )

        st.dataframe(
            schema,
            use_container_width=True
        )

    agent = create_data_analyst(df)

    for message in (st.session_state.conversation_history):
        with st.chat_message(
            message["role"]
        ):
            st.write(
                message["content"]
            )

    question = st.chat_input(
        "Ask a question about your data..."
    )

    if question: 
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing you data..."):
                answer = ask_agent(agent=agent, 
                                conversation_history=st.session_state.conversation_history,
                                question=question)

                st.write(answer)

