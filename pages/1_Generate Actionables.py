## import the CSV file and output the CSV file 

import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import AzureChatOpenAI
from langchain.llms import Banana
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import OutputFixingParser
import csv
import os
import base64
from datetime import datetime

load_dotenv()

chat_llm = AzureChatOpenAI(
    deployment_name="gpt-35-turbo",
    model_name="gpt-35-turbo",
    temperature=0.6
)

llm_options = {
    "gpt-35-turbo": AzureChatOpenAI(
        deployment_name="gpt-35-turbo",
        model_name="gpt-35-turbo",
        temperature=0.6  # Assuming you want to set a default temperature
    ),
    "mistral": Banana(
        model_key="",
        model_url_slug="demo-mistral-7b-instruct-v0-1-lnlnzqkn5a",
    ),
    "llama2-13b": Banana(
        model_key="",
        model_url_slug="llama2-13b-chat-awq-loh5cxk85a",
    )
}

is_dev = os.getenv("IS_DEV", "false").lower() == "true"
data_path = "data" if is_dev else "/data"
sample_data_path = "sample_data"


def get_download_link(df):
    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    file_name = f"Actionables_{current_datetime}.csv"

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{file_name}">Download CSV File</a>'

    return href


def get_prompt_for_action(action):
    """Get the prompt for a given action from prompts.csv."""
    prompts_df = pd.read_csv(os.path.join(sample_data_path, "prompts.csv"))  # Read the CSV containing prompts from the correct path
    prompt_row = prompts_df[prompts_df['Action'] == action]
    if not prompt_row.empty:
        return prompt_row['Prompt'].iloc[0]
    else:
        return None 


def prompt_generator(df,llm):
    #####output parser #############################################

    Action_schema = ResponseSchema(name="Actionable",
                                   description="List of Actionable requirements from the text")

    response_schemas = [Action_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    new_parser = OutputFixingParser.from_llm(parser=output_parser, llm=llm)
    ###########################################################################

    title_template = get_prompt_for_action('Actionable')
    
    prompt = ChatPromptTemplate.from_template(template=title_template)

    ##############################################################################################

    df2 = pd.DataFrame(columns=['Regulatory text', 'Actionable'])

    for index, row in df.iterrows():
        messages = prompt.format_messages(topic=row['Regulatory text'], format_instructions=format_instructions)
        response = chat_llm(messages)
        response_as_dict = new_parser.parse(response.content)

        # Extract 'Actionable' from the response
        actionable = response_as_dict.get('Actionable', '')

        if actionable:
            if isinstance(actionable, list):
                # If 'Actionable' is a list, iterate over its items
                for item in actionable:
                    df2 = df2.append({'Regulatory text': row['Regulatory text'], 'Actionable': item}, ignore_index=True)
            else:
                df2 = df2.append({'Regulatory text': row['Regulatory text'], 'Actionable': actionable},
                                 ignore_index=True)
    st.subheader("Actionables")
    st.dataframe(df2)
    st.markdown(get_download_link(df2), unsafe_allow_html=True)


def main():
    st.image('logo.png')
    st.title("👨‍💻 Extract Actionables")

    # File upload
    file = st.file_uploader("Upload a CSV file", type=["csv"])

    st.markdown("### Download Sample CSV")
    sample = pd.read_csv(os.path.join(sample_data_path, "sample.csv"))
    st.markdown(get_download_link(sample), unsafe_allow_html=True)

    selected_llm = st.selectbox("Select a Language Model", options=list(llm_options.keys()))  

    # Set the LLM based on the selection
    llm = llm_options[selected_llm]

    if file is not None:
        # Read CSV file
        df = pd.read_csv(file)

        # Display preview
        st.subheader("CSV File Preview")
        st.dataframe(df)

        # Button to process the file
        if st.button("Extract Actionable"):
            prompt_generator(df, llm) 


if __name__ == "__main__":
    main()
