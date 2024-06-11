from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import requests
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough

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

chain = RunnablePassthrough.assign(
    text = lambda x: scrape_text(x['url'])[:10000]
) | summary_prompt | ChatOpenAI() | StrOutputParser()

# Main execution
if __name__ == "__main__":
    print(chain.invoke({
        "question": "What is langsmith?",
        "url": url
    }))