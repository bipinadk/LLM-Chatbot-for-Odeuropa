# LLM-Chatbot-for-Odorupa

To get the app running:

REPOSITORY SETUP
- Clone the repository
- In server/odeuropa_chatbot.py, set your OPENAI key in "os.environ["OPENAI_API_KEY"] = "{enter key here}""
- Currently, the "test_clean_vectordb" vector database is present in the repository because of the size of the full repository
- For the current version of the repository, only the test vector db is available outside of Eurecom servers
- If you have your own vector database, copy it to /server and add the path to "Chroma(persist_directory="{path_to_vectordb}", embedding_function=embedding_function)" in server/odeuropa_chatbot.py

RUNNING SERVER
- Open two terminals, one in /server and one in /frontend/odeuropa-chatbot-react
- In /server, run "python app.py" to start the server
- In /frontend/odeuropa-chatbot-react, run "npm install", wait for packages to install, then run "npm start" to start the website

INTERACTING
- Now you can chat with the bot on the website's chat interface, receive replies and see sources related to replies
- Currently, some LLM feedback is logged in the server terminal, and some commented print statements are present in the server code which can be uncommented for more logging
