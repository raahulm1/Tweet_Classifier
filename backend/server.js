const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = 5000;

// Middleware
app.use(cors());
app.use(bodyParser.json());

// Route to handle tweet classification
app.post('/classify', async (req, res) => {
    const { tweet } = req.body;

    if (!tweet) {
        return res.status(400).json({ error: 'Tweet is required' });
    }

    try {
        // Send the tweet to the Python Flask API
        const flaskResponse = await axios.post('http://127.0.0.1:8000/predict', { tweet });

        // Send Flask's response back to the client
        res.json({ result: flaskResponse.data.result });
    } catch (error) {
        console.error('Error communicating with Flask API:', error.message);
        res.status(500).json({ error: 'Error communicating with the prediction server' });
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
