import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

def create_data_analyst(df):
    """
    Create an AI Data Aalyst for the provided DataFrame.
    """

    @tool
    def get_data_summary()-> str:
        """
        Return basic information about the dataset, including number of rows, columns and the data types.
        """

        summary =[]
        summary.append(
            f"Number of rows: {len(df)}"
        )

        summary.append(
            f"Number of Columns: {len(df.columns)}"
        )

        summary.append(
            f"Columns: {', '.join(df.columns.tolist())}"
        )

        summary.append(
            "Data types: "
        )

        for column, dtype in df.dtypes.items():

            summary.append(
                f"- {column}: {dtype}"
            )

        return "\n".join(summary)

    @tool
    def get_column_statistics(column: str)->str:
        """
        Return statistical information for a numeric column.

        Statistics include count, mean, minimum, maximum, and standard deviation.
        """

        if column not in df.columns:
            return (
                f"Column '{column}' does not exist"
            )

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            return(
                f"Column '{column}' is not numeric"
            )

        result ={
            "count": int(df[column].count()),
            "mean":float(df[column].mean()),
            "minimum": float(df[column].min()),
            "maximum": float(df[column].max()),
            "standard_deviation": float(df[column].std())
        }

        return str(result)

    @tool
    def filter_data(column:str, operator: str, value: str)-> str:
        """
        Filter the dataset using a column, comparison operator and the value.

        Supported operator: 
        >, <, >=, <=, == 
        """

        if column not in df.columns:
            return (
                f"Column '{column}' does not exist"
            )

        try:
            if pd.api.types.is_numeric_dtype(
                df[column]
            ):
                value = float(value)

                if operator == ">":
                    result = df[df[column]>value]

                elif operator == "<":
                    result = df[df[column]<value]

                elif operator == ">=":
                    result = df[df[column]>=value]

                elif operator == "<=":
                    result = df[df[column]<=value]

                elif operator == "==":
                    result = df[df[column]==value]

                else:
                    return(
                        "unsupported operator. "
                        "Use >, <, >=, <=, =="
                    )

                if result.empty:
                    return "No matching records found."

                return result.head(20).to_string(index=False)


        except Exception as e:
            return (
                f"Error while fetching data: {e}"
            )


    @tool
    def aggregate_data(
        column: str, operation: str
    ) -> str:
        """
        Perform an aggregation operation on a numeric column.

        Supported operations:
        sum, mean, min, max and count.
        """

        if column not in df.columns:
            return (
                f"Column '{column}' does not exist"
            )

        if not pd.api.types.is_numeric_dtype(
            df[column]
        ):
            return(
                f"Column '{column}' is not numeric"
            )

        if operation == "sum":
            result = df[column].sum()

        elif operation =="mean":
            result = df[column].mean()

        elif operation =="min":
            result = df[column].min()

        elif operation =="max":
            result = df[column].max()

        elif operation =="count":
            result = df[column].count()

        else:
            return (
                "unsupported operation."
                "use sum, mean, max, min, or count"
            )

        return (
            f"{operation} of {column}"
            f"{result:,.2f}"
        )

    @tool
    def group_and_aggregate(
        group_column: str,
        value_column: str,
        operation: str
    )-> str:
        """
        Group the dataset by one column and perform an aggregation on another numeric column.

        Supported operations:
        sum, mean, min, max, and count.
        """

        if group_column not in df.columns:
            return(
                f"Column '{group_column} does not exist."
            )

        if value_column not in df.columns:
            return(
                f"Column '{value_column} does not exist."
            )

        if not pd.api.types.is_numeric_dtype(
            df[value_column]
        ):
            return(
                f"Column {value_column} must be a numeric column"
            )

        if operation not in ["sum", "min", "mean", "max", "count"]:
            return "unsupported operation."

        result = (
            df.groupby(group_column)[value_column].agg(operation).sort_values(
                ascending=False
            ).head(20)
        )

        return result.to_string()

    tools = [
        get_data_summary,
        get_column_statistics,
        filter_data,
        aggregate_data,
        group_and_aggregate
    ]

    model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite")

    agent = create_agent(
        model=model,
        tools=tools,

        system_prompt="""
        You are an AI Data Analyst.

        You answer questions about the uploaded CSV dataset.

        IMPORTANT RULES:
        1. Use the available tools whenever the calculations or data analysis is required.
        2. Do not perform numerical calulations by yourself when a tool can perform them.
        3. Only use the columns that actually exist in the dataset.
        4. Never invent data.
        5. Clearly explain the results returned by the tools.
        6. If the question cannot be answered using the available dataset, explain that clearly. 
"""
    )

    return agent


def ask_agent(
        agent,
        conversation_history,
        question
):
    """
    Send a question to the agent while maintaining the conversation history.
    """

    conversation_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    response = agent.invoke({
        "messages": conversation_history
    })

    answer = response["messages"][-1].text

    conversation_history.append(
        {
            "role":"assistant",
            "content": answer
        }
    )

    return answer