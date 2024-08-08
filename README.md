# pebblo-hackathons
		News Article summarizer with Q&A bot
Overview
This project is a Flask web application that summarizes newspaper articles and answers questions about them. The application uses web scraping to fetch article content, natural language processing (NLP) to remove stopwords, and a combination of BERT for question-answering and the LangChain framework for document retrieval and language model integration.
Features
•	Summarize Articles: Enter a URL of a newspaper article to get a concise summary.
•	Ask Questions: Enter questions about the summarized article to get answers.
Prerequisites
•	Python 3.x
•	Flask
•	Requests
•	BeautifulSoup4
•	Transformers (Hugging Face)
•	nltk
•	langchain_community
•	langchain_text_splitters
•	langchain_core
Setup Instructions
1.	Install the required libraries:
pip install flask requests beautifulsoup4 transformers nltk langchain_community langchain_text_splitters langchain_core.
2.	Ensure index.html is in the templates folder and all files are in the same directory:
project-directory/
├── app.py
└── templates/
    └── index.html
3.	Run the Flask application:
python app.py
4.	Open the generated link in your web browser.
Sample outputs:
1.1  User has to paste the link in the given tab below

 
1.2   After clicking on Get summary, the summary will be generated.
 
1.3    User can ask the questions about the article and can interact with the chat bot. 

1.4    To end the session user can enter ‘exit’  
