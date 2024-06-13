## Research Assistant

### Project Introduction

This project aims to create a highly efficient research assistant using LangChain and OpenAI.
Inspired by the Tavil AI platform, this project focuses on developing a complex application that goes beyond simple chat
functionalities.
The assistant performs extensive web research by generating research questions, fetching relevant web pages, and
summarizing the information into comprehensive reports.
This approach involves multiple decision points and optimizes the process through parallelization, ensuring better and
more interesting responses.
The end goal is to build a versatile research assistant capable of performing detailed and time-intensive research tasks
effectively.

### Prerequisites

- Python 3.11
- pip (Python package installer)
- Git (optional)

### Step 1: Initial Setup

#### 1. Initialize the Environment

First, let's set up the environment and install necessary dependencies.

1. **Create a `.env` file:**
    - This file will store your API keys and other configuration settings. Ensure it is included in your `.gitignore`
      file to prevent it from being committed to your repository.

   Example `.env` file:
   ```plaintext
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
   LANGCHAIN_API_KEY="your_langchain_api_key"
   LANGCHAIN_PROJECT="ResearchAssistant"
   OPENAI_API_KEY="your_open_api_key"
   ```

2. **Install required packages:**
   ```bash
   pip install langchain langchain_community openai streamlit python-dotenv
   ```
   ```bash
   pip install -U langchain-cli
   ```

### Step 2: Setup LangServe and LangSmith

#### 1. LangServe Setup

Set up LangServe to manage our application deployment.
Use the LangServe CLI to create a new application called `research-assistant`.

```bash
langchain app new research-assistant
```

#### 2. LangSmith Setup

Make sure u have created a LangSmith project for this lab.

**Project Name:** ResearchAssistant

### Step 3: Implement the Web Scraping and Summarization Chain

In this step, we will implement a chain that integrates web scraping to fetch relevant content and summarizes the
information based on research questions.

#### 1. Install Required Packages

To perform web scraping and handle HTTP requests, we need to install a few additional packages.

```bash
pip install beautifulsoup4
```

```bash
pip install requests
```

#### 2. Create the Summarization Chain

Add a chain that uses `ChatPromptTemplate` and `ChatOpenAI` for summarizing questions based on web-scraped context.

**File**: `research-assistant/app/chain.py`

**Code for `chain.py`:**

```python
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
```

This script sets up the summarization chain and includes a function to scrape text **from a given URL**. It integrates
web scraping using BeautifulSoup to retrieve and parse page content.

#### 3. Test the Chain

Run the `chain.py` file and inspect the results in LangSmith to ensure that the chain is functioning correctly.

<img src="https://i.imghippo.com/files/mEwGC1718102113.jpg" alt="" border="0">

This output indicates that the chain is working correctly and generating a summarized response based on the context
provided by the scraped web page.

#### Key Concepts

##### 1. DuckDuckGo Search API

- **Definition**: DuckDuckGo Search API is a tool that allows developers to access DuckDuckGo search results
  programmatically. It provides an easy way to integrate web search functionality into applications.
- **Usage**: It is used in this project to perform web searches and retrieve links to relevant web pages based on a
  user's query.

##### 2. BeautifulSoup

- **Definition**: BeautifulSoup is a Python library used for parsing HTML and XML documents. It creates a parse tree for
  parsing HTML and XML documents to extract data from HTML, which is useful for web scraping.
- **Usage**: BeautifulSoup is typically used in conjunction with requests to fetch and parse web pages. It allows you to
  navigate the parse tree and search for specific elements, such as tags, attributes, and text.
- **Example**:
  ```python
  from bs4 import BeautifulSoup

  response = requests.get('https://example.com')
  soup = BeautifulSoup(response.text, 'html.parser')
  print(soup.get_text())
  ```

##### 3. Requests

- **Definition**: Requests is a simple and elegant HTTP library for Python, built for human beings.
- **Usage**: It is used to send HTTP requests to fetch web pages or other resources.
- **Example**:
  ```python
  import requests

  response = requests.get('https://example.com')
  print(response.text)
  ```

### Step 4: Integrate DuckDuckGo Web Search with Text Scraping and Summarization

In this step, we will integrate DuckDuckGo web search with the text scraping and summarization functionality to
dynamically fetch URLs based on a query and summarize the content.

#### 1. Install Required Packages

```bash
pip install -U duckduckgo-search
```

#### 2. Update `chain.py` to Include DuckDuckGo Web Search

**File**: `research-assistant/app/chain.py`

**Updated Code for `chain.py`:**

```python
...

from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

...

RESULT_PER_QUESTION = 3

ddg_search = DuckDuckGoSearchAPIWrapper()

...

scrape_and_summarize_chain = RunnablePassthrough.assign(
    text = lambda x: scrape_text(x["url"])[:10000]
) | summary_prompt | ChatOpenAI() | StrOutputParser()

chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

# Main execution
if __name__ == "__main__":
    print(chain.invoke({
        "question": "What is LangSmith?"
    }))
```

<img src="https://i.imghippo.com/files/Af6oR1718104809.jpg" alt="" border="0">

This script enhances the previous functionality by integrating DuckDuckGo web search. It fetches relevant URLs based on
the query, scrapes the text content from those URLs, and then summarizes the information.

#### Detailed Explanation

1. **Performing Web Searches**:
    - The DuckDuckGo web search functionality is used to perform an internet search based on the user’s query. This
      means that instead of relying on predefined URLs or static content, the research assistant dynamically searches
      the web for the most relevant and up-to-date information.
    - The search results are limited to a specific number (defined by `RESULT_PER_QUESTION = 3`), ensuring that only the
      top 3 most relevant web pages are considered. This focuses the research on high-quality sources and keeps the
      processing manageable.

2. **Scraping Web Page Content**:
    - Once the URLs are retrieved from the DuckDuckGo search, the next step is to scrape the text content from these web
      pages. This involves fetching the HTML content of each URL and extracting the readable text using BeautifulSoup.
    - By scraping the web page content, the research assistant gathers detailed information that can be used to answer
      the user’s query. This step ensures that the assistant has access to the full text of the relevant articles or web
      pages.

3. **Summarizing the Retrieved Content**:
    - After scraping the text content, the assistant uses the LangChain summarization pipeline to generate concise
      summaries of the information. This involves feeding the scraped text into a predefined prompt template, which
      instructs the language model to summarize the text and answer the user’s question.
    - The summarization step is crucial for distilling large amounts of text into useful insights, making the
      information more accessible and easier to understand.

#### Key Concepts

##### 1. DuckDuckGo Search API

- **Definition**: DuckDuckGo Search API is a tool that allows developers to access DuckDuckGo search results
  programmatically. It provides an easy way to integrate web search functionality into applications.
- **Usage**: It is used in this project to perform web searches and retrieve links to relevant web pages based on a
  user's query.
- **Example**:
  ```python
  from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

  ddg_search = DuckDuckGoSearchAPIWrapper()
  results = ddg_search.results("What is LangSmith?", 3)
  print(results)
  ```

#### 2. Test the Chain

Run the `chain.py` file and inspect the results in LangSmith to ensure that the chain is functioning correctly.

**Steps to Test:**

1. Run the `chain.py` file:
   ```bash
   python research-assistant/app/chain.py
   ```
2. Go to LangSmith to inspect the results and ensure that the summarization is accurate and based on the web-scraped
   content.

#### Example Output

After running the script, you should see output similar to the following:

<img src="https://i.imghippo.com/files/qat011718104686.jpg" alt="" border="0">

This output indicates that the chain is working correctly and generating a summarized response based on the context
provided by the scraped web page.

### Step 5: Add Search Question Chain for Generating Search Queries

In this step, we will add a chain that generates search queries based on the input question using `ChatPromptTemplate`
and `ChatOpenAI`.

#### 1. Update `chain.py` to Include Search Question Chain

**File**: `research-assistant/app/chain.py`

**Updated Code for `chain.py`:**

```python
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
```

<img src="https://i.imghippo.com/files/PSeDI1718107319.jpg" alt="" border="0">
This script adds a new chain that generates search queries based on the input question. It integrates the search question chain with the existing functionality for web scraping and summarization.

#### 2. Test the Search Question Chain

Run the `chain.py` file and inspect the results to ensure that the search question chain is generating appropriate
search queries.
After running the script, you should see output similar to the following:

<img src="https://i.imghippo.com/files/lf2ld1718106946.jpg" alt="" border="0">

This output indicates that the search question chain is working correctly and generating relevant search queries based
on the input question.
By following these steps, you can ensure that your research assistant is capable of generating search queries and
summarizing information effectively.

### Step 6: Integrate Search Query Chain with Web Search and Summary Chains

In this step, we will integrate the search query chain with the web search and summary chains to handle multiple search
queries and summarize the results.

#### 1. Update `chain.py` to Integrate Search Query Chain

**File**: `research-assistant/app/chain.py`

**Updated Code for `chain.py`:**

```python
...
web_search_chain = RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
) | (lambda x: [{"question": x["question"], "url": u} for u in x["urls"]]) | scrape_and_summarize_chain.map()

chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()

# Main execution
if __name__ == "__main__":
    print(chain.invoke({
        "question": "What is the different between LangChian and LangSmith?"
    }))
```

<img src="https://i.imghippo.com/files/vtY6U1718107926.jpg" alt="" border="0">
This script integrates the search question chain with the web search and summary chains, allowing for multiple search queries to be generated and their results summarized.

#### 2. Test the Integrated Chain

Run the `chain.py` file and inspect the results to ensure that the integrated chain is generating appropriate search
queries, performing web searches, and summarizing the content.
<img src="https://i.imghippo.com/files/06cBS1718108002.jpg" alt="" border="0">
This output indicates that the integrated chain is working correctly, generating relevant search queries, performing web
searches, and summarizing the content effectively.

### Step 7: Add Full Research Chain with Comprehensive Report Generation

In this step, we will integrate the search query generation, web search, and summarization into a full research chain
capable of generating comprehensive research reports.

#### 1. Update `chain.py` to Include Full Research Chain

**File**: `research-assistant/app/chain.py`

**Updated Code for `chain.py`:**

```python
...

full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | web_search_chain.map()


def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)


writer_system_prompt = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."

research_prompt_template = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- 
The report should focus on the answer to the question, should be well structured, informative, 
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", writer_system_prompt),
        ("user", research_prompt_template),
    ]
)

chain = RunnablePassthrough.assign(
    research_summary = full_research_chain | collapse_list_of_lists
) | prompt | ChatOpenAI() | StrOutputParser()

# Main execution
if __name__ == "__main__":
    print(chain.invoke({
        "question": "What is the difference between LangChain and LangSmith?"
    }))
```

<img src="https://i.imghippo.com/files/518P91718148307.jpg" alt="" border="0">
This script integrates all previous components into a full research chain capable of generating comprehensive research reports based on the summarized content.

#### 2. Test the Full Research Chain

Run the `chain.py` file and inspect the results to ensure that the full research chain is generating comprehensive
research reports based on the summarized content.

<img src="https://i.imghippo.com/files/AUftT1718148332.jpg" alt="" border="0">

This output indicates that the full research chain is working correctly, generating a comprehensive report based on the
summarized content from web searches.

### Step 8: Enhance Scrape and Summarize Chain to Include URL in Summary Output

In this step, we will enhance the `scrape_and_summarize_chain` to include the URL in the final summary output.

#### 1. Update `chain.py` to Enhance Scrape and Summarize Chain

**File**: `research-assistant/app/chain.py`

Modify the `scrape_and_summarize_chain` to include the URL in the summary output.

**Updated Code for `chain.py`:**

```python
scrape_and_summarize_chain = RunnablePassthrough.assign(
    summary = RunnablePassthrough.assign(
        text = lambda x: scrape_text(x["url"])[:10000]
    ) | summary_prompt | ChatOpenAI() | StrOutputParser()
) | (lambda x: f"URL: {x['url']}\n\nSUMMARY: {x['summary']}")
```

This enhancement ensures that each summary includes the URL of the web page it was generated from, providing a clear
reference for the source of the information.

#### 2. Test the Enhanced Chain

Run the `chain.py` file and inspect the results to ensure that the summaries now include the URL of the source.

<img src="https://i.imghippo.com/files/vZczR1718149119.jpg" alt="" border="0">

### Step 9: Integrate ArxivRetriever for Document-Based Summaries

In this step, we will integrate the ArxivRetriever to fetch academic papers and process them for document-based
summaries.
We will also differentiate between web-based research (using `web_chain.py`) and document-based research (
using `doc_chain.py`).

#### 1. Install Required Package

To use ArxivRetriever, install the necessary package:

```bash
pip install arxiv
```

#### 2. Update `web_chain.py` and Create `doc_chain.py`

**1. Rename `chain.py` to `web_chain.py`**

Rename the existing `chain.py` to `web_chain.py` to indicate that it handles web-based research.

**2. Create `doc_chain.py` for Document-Based Research**

Create a new file `doc_chain.py` to handle document-based research using ArxivRetriever.

**File**: `research-assistant/app/doc_chain.py`

Differences between `web_chain.py`
<img src="https://i.imghippo.com/files/13QO11718152604.jpg" alt="" border="0">
<img src="https://i.imghippo.com/files/QKGRJ1718152702.jpg" alt="" border="0">

#### Key Concepts

##### 1. ArxivRetriever

- **Definition**: ArxivRetriever is a tool that allows developers to fetch academic papers from the Arxiv repository
  programmatically. It provides an easy way to access and summarize research papers.
- **Usage**: It is used in this project to perform document-based research and retrieve summaries of academic papers
  based on a user's query.
- **Example**:
  ```python
  from langchain.retrievers import ArxivRetriever

  retriever = ArxivRetriever()
  docs = retriever.get_summaries_as_docs("What papers did Emil Khalisi write?")
  print(docs)
  ```

#### 2. Test the Document-Based Research Chain

Run the `doc_chain.py` file and inspect the results to ensure that the document-based research chain is functioning
correctly and generating summaries of academic papers.

<img src="https://i.imghippo.com/files/6trR21718152471.jpg" alt="" border="0">

### Step 10: Serve the Application Using LangServe

#### 1. Update `server.py`:

<img src="https://i.imghippo.com/files/ZM7CQ1718153516.jpg" alt="" border="0">

#### 2. Update `web_chain.py` and `doc_chain.py`:

<img src="https://i.imghippo.com/files/fNpPj1718153555.jpg" alt="" border="0">

#### 3. Serving the Application by LangServe

Run the following commands to set up and serve the application using LangServe.

   ```bash
   cd research-assistant
   langchain serve
   ```

You can now access the application through the following links:

Access [Web Search Playground](http://127.0.0.1:8000/research-assistant-web_chain/playground/)

Access [Arxiv Search Playground](http://127.0.0.1:8000/research-assistant-doc_chain/playground/)

<img src="https://i.imghippo.com/files/fUpIL1718153608.jpg" alt="" border="0">
<img src="https://i.imghippo.com/files/9DJ481718153627.jpg" alt="" border="0">