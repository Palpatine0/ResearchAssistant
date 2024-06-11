from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import json

from dotenv import load_dotenv
from langchain.callbacks.tracers.langchain import wait_for_all_tracers

load_dotenv()
wait_for_all_tracers()


def scrape_text(url: str):
    """
    Scrape text from web
    """
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            page_text = soup.get_text(separator = " ", strip = True)
            return page_text
        else:
            return f"Failed to retrieve the webpage: Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrieve the webpage: {e}"


url = "https://blog.langchain.dev/announcing-langsmith/"

RESULT_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()


def web_search(query: str, num_results: int = RESULT_PER_QUESTION):
    results = ddg_search.results(query, num_results)
    return [r["link"] for r in results]


# Text summary pipeline
summary_template = """
{text} 
-----------
Using the above text, answer in short the following question: 
> {question}
-----------
if the question cannot be answered using the text, imply summarize the text. 
Include all factual information, numbers, stats etc if available.
"""
summary_prompt = ChatPromptTemplate.from_template(summary_template)

scrape_and_summarize_chain = RunnablePassthrough.assign(
    text = lambda x: scrape_text(x["url"])[:10000]
) | summary_prompt | ChatOpenAI() | StrOutputParser()

# Search pipeline
search_question_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

search_question_chain = search_question_prompt | ChatOpenAI(temperature = 0) | StrOutputParser() | json.loads

chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

# Main execution
if __name__ == "__main__":
    print(search_question_chain.invoke({
        "question": "What is the different between LangChian and LangSmith?"
    }))
