import React, { useState } from 'react';
import './App.css';

function App() {
  const [userMessage, setUserMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [sources, setSources] = useState([]);

  const sendMessage = async () => {

    if (userMessage.trim() === '') {
      return;
    }
    
    setUserMessage('');
    try {
      const response = await fetch('http://localhost:5000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          chat_history: messages,
          question: userMessage,
        }),
      });

      if (response.ok) {
        const data = await response.json();
        const newMessage = { from: 'user', text: userMessage };
        setMessages((prevMessages) => [...prevMessages, newMessage]);
        const newReply = { from: 'bot', text: data.answer };
        setMessages((prevMessages) => [...prevMessages, newReply]);
        setSources(data.sources);
       
      } else {
        console.error('Failed to get response from server:', response.statusText);
      }

    } catch (error) {
      console.error('Error sending message:', error);
    }

  };

  const handleInputChange = (e) => {
    setUserMessage(e.target.value);
  };

  const handleSubmit = (e) => {
    e.preventDefault()
    sendMessage()
  }

  return (
    <div className="App">
      <div className='Title-Container'>
        <h1 className='Title'>
          Odeuropa Chatbot
        </h1>
      </div>
      <div className="chat-window">
        {messages.map((message, index) => (
          <div key={index} className={message.from === 'user' ? 'user-message' : 'bot-message'}>
            {message.text}
            {message.from === 'bot' && <div style={{margin: '10px'}}/>}
            {message.from === 'bot' && (
              sources.map((source, index) => (
                <div key={index}>
                  Source {index + 1}:&nbsp;
                  <a href={source} target="_blank" rel="noopener noreferrer">
                    {source}
                  </a>
                </div>
              ))
            )}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="ChatInput">
        <input
          type="text"
          placeholder="Type your message..."
          value={userMessage}
          onChange={handleInputChange}
        />
        <button type="submit">Send</button>
      </form>
    </div>
  );
}

export default App;
