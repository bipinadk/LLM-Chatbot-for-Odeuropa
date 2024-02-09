from flask import Flask, jsonify, request
from flask_cors import CORS
from odeuropa_chatbot import chat, parse_metadatas, setup_llm, setup_vectordb

app = Flask(__name__)
CORS(app)

with app.app_context():
    setup_vectordb(app)
    setup_llm(app)

@app.route('/api/message', methods=['GET'])
def get_message():
    return jsonify({'message': 'Hello from the Python server!'})

@app.route('/chat', methods=['POST'])
def answer_question():
    try:
        data = request.get_json()

        if 'question' in data and 'chat_history' in data:
            question = data['question']
            chat_history = data['chat_history']
            
            metadatas, new_history, answer = chat(app, chat_history, question)
            print("got the chat reply")
            #print(metadatas)
            #print(new_history)
            #print(answer)
            
            processed_metadatas = parse_metadatas(metadatas)
            print("got the processed md")
            #print(processed_metadatas)
            sources = []
            for md in processed_metadatas:
                sources.append(md['idlink'])

            response = jsonify({'answer': answer, 'chat_history': new_history, 'sources': sources})
            response.status_code = 200
            return response
        else:
            response = jsonify({'status': 'error', 'message': 'Invalid request'})
            response.status_code = 400
            return response

    except Exception as e:
        response = jsonify({'status': 'error', 'message': str(e)})
        response.status_code = 400
        return response

if __name__ == '__main__':
    app.run(port=5000)