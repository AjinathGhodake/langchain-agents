from dotenv import load_dotenv
from tools.tools import get_profile_url_tavily

from langchain_openai import ChatOpenAI

from langchain_community.llms import Ollama
from langchain.prompts.prompt import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor,
)
from langchain import hub
import os

load_dotenv()


def lookup(name: str) -> str:

    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o",
        openai_api_key=os.environ["OPENAI_API_KEY"],
    )
    # llm = Ollama(model="llama3", temperature=0)

    template = """given the name {name_of_person} I want you to find a link to their\
        twitter profile page, and extract from it their username. In your\
            In Your Final answer only the person's username"""
    prompt_template = PromptTemplate(
        template=template,
        input_variables=["name_of_person"],
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google 4 twitter profile page",
            func=get_profile_url_tavily,
            description="useful for when you need get the twitter Page URL",
        ),
    ]
    react_prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(
        llm=llm, tools=tools_for_agent, prompt=react_prompt
    )
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_agent,
        verbose=True,
        handle_parsing_errors=True,
    )
    result = agent_executor.invoke(
        input={"input": prompt_template.format_prompt(name_of_person=name)}
    )
    linked_profile_url = result["output"]
    return linked_profile_url


if __name__ == "__main__":
    twitter_url = lookup(name="Elon Musk")
    print(twitter_url)
