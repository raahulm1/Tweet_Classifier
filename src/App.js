import React, { useState } from 'react';
import './App.css'; // Import the CSS file

function App() {
  const [tweet, setTweet] = useState('');
  const [result, setResult] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    const response = await fetch('http://localhost:5000/classify', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ tweet }),
    });
    const data = await response.json();
    setResult(data.result);
  };

  return (
    <div style={{ padding: '20px' }}>
      <h1>TweetClassify</h1>
      <form onSubmit={handleSubmit}>
        <textarea
          value={tweet}
          onChange={(e) => setTweet(e.target.value)}
          placeholder="Enter your tweet here"
          rows="4"
          cols="50"
        />
        <br />
        <button type="submit">Classify</button>
      </form>
      {result && (
        <div className="result">
          <h2>Result:</h2>
          <p>{result}</p>
        </div>
      )}
      <footer>
        <p>Made by <strong>Rahul Mandalageri</strong></p>  
        <div className="social-icons">
          <a href="https://www.linkedin.com/in/rahul-mandalageri-05977b253/" target="_blank" rel="noopener noreferrer">
            <i className="fab fa-linkedin"></i>
          </a>
          <a href="https://github.com/raahulm1" target="_blank" rel="noopener noreferrer">
            <i className="fab fa-github"></i>
          </a>
          <a href="https://www.instagram.com/rahulm_12_/?hl=en" target="_blank" rel="noopener noreferrer">
            <i className="fab fa-instagram"></i>
          </a>
        </div>
      </footer>
    </div>
  );
}

export default App;
