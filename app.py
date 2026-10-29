import os
from datetime import datetime
from pathlib import Path
from load_dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

import streamlit as st
from crewai import Agent, Crew, Process, Task, LLM
from crewai.tools import BaseTool
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
# Defaults target the local Docker Model Runner setup.
# To use a hosted OpenAI model instead, set MODEL and
# OPENAI_API_KEY, and remove/leave blank LLM_BASE_URL.
MODEL = os.getenv("MODEL", "ai/qwen3-vl:2B-BF16")
LLM_BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "http://localhost:12434/engines/v1",
)
OUTPUT_DIR = Path(os.getenv("SUPPORT_OUTPUT_DIR", "support_logs"))


def build_llm():
    """Build a CrewAI LLM from environment configuration."""
    if LLM_BASE_URL:
        return LLM(
            model=MODEL,
            base_url=LLM_BASE_URL,
            api_key=os.getenv("LOCAL_LLM_API_KEY", "not-needed"),
            custom_openai=True,
        )

    # Hosted model path. The key is read from the environment.
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "Set OPENAI_API_KEY for a hosted model, or configure LLM_BASE_URL "
            "for an OpenAI-compatible local model."
        )
    return LLM(model=MODEL, api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------
# Custom tool: Entry Agent writes the support record to disk
# ---------------------------------------------------------
class SaveSupportRecordInput(BaseModel):
    content: str = Field(
        ...,
        description="Complete support record containing the query and both answers.",
    )
    filename: str = Field(
        default="support_record.txt",
        description="Name for the text file. Use a safe .txt filename.",
    )


class SaveSupportRecordTool(BaseTool):
    name: str = "save_support_record"
    description: str = (
        "Save the complete customer-support query and both agent answers "
        "to a plain-text .txt file. Pass the entire record in the content field."
    )
    args_schema: type[BaseModel] = SaveSupportRecordInput

    def _run(self, content: str, filename: str = "support_record.txt") -> str:
        # Prevent path traversal and force a .txt extension.
        safe_name = Path(filename).name
        if not safe_name.lower().endswith(".txt"):
            safe_name = f"{safe_name}.txt"

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        path = OUTPUT_DIR / safe_name
        path.write_text(content, encoding="utf-8")
        return f"Support record saved successfully: {path.resolve()}"


# ---------------------------------------------------------
# Build the three agents and sequential tasks
# ---------------------------------------------------------
def run_support_crew(query: str):
    if not os.getenv("SERPER_API_KEY"):
        raise ValueError(
            "SERPER_API_KEY is missing. Set it in your environment before running the app."
        )

    llm = build_llm()
    web_search_tool = SerperDevTool()
    save_record_tool = SaveSupportRecordTool()

    # Agent 1: answer from its own knowledge; no tools.
    assistant = Agent(
        role="Assistant",
        goal="Answer the customer's query directly and clearly using your own knowledge.",
        backstory=(
            "You are a helpful customer-support assistant. Give a direct, accurate, "
            "self-contained answer. Do not claim to have searched the web."
        ),
        tools=[],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # Agent 2: searches the web and creates a separate evidence-based answer.
    web_assistant = Agent(
        role="Web Search Assistant",
        goal="Research the customer's query online and answer using relevant web results.",
        backstory=(
            "You are a careful web research assistant. Search for relevant, current "
            "information, prioritize trustworthy sources, and distinguish verified "
            "facts from uncertainty. Include useful source URLs in your answer."
        ),
        tools=[web_search_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    # Agent 3: receives both earlier task outputs and saves the record.
    entry_agent = Agent(
        role="Entry Agent",
        goal="Save the customer's query and both answers into a plain-text file.",
        backstory=(
            "You maintain accurate customer-support records. Preserve the original "
            "query and both answers without changing their meaning. Use the provided "
            "save_support_record tool to write the record to disk."
        ),
        tools=[save_record_tool],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    assistant_task = Task(
        description=(
            "Answer this customer query directly from your own knowledge. "
            "Do not use web search or invent sources.\n\nCustomer query:\n{query}"
        ),
        expected_output=(
            "A clear, direct answer to the customer query. Do not include web citations "
            "unless they are already known and necessary."
        ),
        agent=assistant,
    )

    web_task = Task(
        description=(
            "Search the web for the customer query, then write an independent answer "
            "based on the search results. Include source URLs where useful. Be transparent "
            "if reliable information is unavailable.\n\nCustomer query:\n{query}"
        ),
        expected_output=(
            "A useful answer based on web research, with relevant source URLs and any "
            "important caveats."
        ),
        agent=web_assistant,
        context=[assistant_task],
    )

    entry_task = Task(
        description=(
            "Create a plain-text support record for the following customer query.\n\n"
            "Customer query:\n{query}\n\n"
            "Use the outputs provided in your task context to capture both the Assistant "
            "answer and the Web Search Assistant answer. Do not omit either answer and "
            "do not invent or rewrite their content. Call the save_support_record tool "
            "with the complete record in this format:\n\n"
            "CUSTOMER SUPPORT RECORD\n"
            "Timestamp: [current timestamp]\n"
            "Customer query:\n[query]\n\n"
            "ASSISTANT ANSWER:\n[answer 1]\n\n"
            "WEB SEARCH ANSWER:\n[answer 2]\n"
        ),
        expected_output=(
            "Confirmation that the query and both complete answers were saved to a .txt file, "
            "including the saved file path."
        ),
        agent=entry_agent,
        context=[assistant_task, web_task],
    )

    crew = Crew(
        agents=[assistant, web_assistant, entry_agent],
        tasks=[assistant_task, web_task, entry_task],
        process=Process.sequential,
        verbose=False,
    )

    result = crew.kickoff(inputs={"query": query})

    # CrewAI exposes each task output in task order. Return the first two answers
    # separately so Streamlit can display them independently.
    outputs = getattr(result, "tasks_output", [])
    answer1 = str(outputs[0].raw) if len(outputs) > 0 else ""
    answer2 = str(outputs[1].raw) if len(outputs) > 1 else ""
    entry_confirmation = str(outputs[2].raw) if len(outputs) > 2 else str(result.raw)

    return answer1, answer2, entry_confirmation


# ---------------------------------------------------------
# Streamlit user interface
# ---------------------------------------------------------
st.set_page_config(page_title="Customer Support Crew", page_icon="💬", layout="wide")
st.title("💬 Customer Support Assistant")
st.write(
    "Ask a question and receive two answers: one from the Assistant's own knowledge "
    "and another based on web research. The Entry Agent saves the query and both answers "
    "to a text file."
)

with st.sidebar:
    st.subheader("Configuration")
    st.caption(f"Model: `{MODEL}`")
    st.caption(
        "Local model endpoint configured"
        if LLM_BASE_URL
        else "Hosted model configuration"
    )
    st.info("API keys are read from environment variables, not from the UI.")

with st.form("support_query_form"):
    query = st.text_area(
        "What can we help you with?",
        placeholder="Type your customer-support question here...",
        height=120,
    )
    submitted = st.form_submit_button("Get answers", type="primary")

if submitted:
    if not query.strip():
        st.warning("Please enter a query before submitting.")
    else:
        try:
            with st.spinner(
                "The Assistant is answering, the Web Search Assistant is researching, "
                "and the Entry Agent is saving the record. This may take a little while..."
            ):
                answer1, answer2, entry_confirmation = run_support_crew(query.strip())

            left, right = st.columns(2)
            with left:
                st.subheader("1. Assistant answer")
                st.write(answer1 or "The Assistant did not return an answer.")

            with right:
                st.subheader("2. Web Search Assistant answer")
                st.write(answer2 or "The Web Search Assistant did not return an answer.")

            st.subheader("3. Entry Agent")
            st.write(entry_confirmation)

            st.success("Crew execution finished.")
            st.caption(
                "Text records are saved in the support_logs folder by default. "
                "Set SUPPORT_OUTPUT_DIR to choose another location."
            )

        except Exception as exc:
            st.error(f"Could not complete the support workflow: {exc}")
            st.info(
                "Check that your local model endpoint is running, the model ID is correct, "
                "and SERPER_API_KEY is set in the same terminal session."
            )
