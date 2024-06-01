from langchain.prompts import PromptTemplate

# from langchain.chat_models import ChatOpenAI
from langchain_community.llms import Ollama
from langchain.chains import LLMChain

from third_parties.linkedin import scrape_linked_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent


def ice_break_with(name: str) -> str:
    linkedin_username = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linked_profile(
        linkedin_profile_url=linkedin_username, mock=True
    )
    summary_template = """
    given the information {information} about a person from I want you to \
    create:
    1. a short summary
    2. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    llm = Ollama(model="llama3", temperature=0)
    chain = LLMChain(llm=llm, prompt=summary_prompt_template)
    res = chain.invoke(input={"information": linkedin_data})

    print(res)


if __name__ == "__main__":
    print("Ice Breaker Enter")
    ice_break_with(name="Shrutika Kamble")
