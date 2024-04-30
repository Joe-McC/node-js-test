// server.js
const express = require('express');
const fs = require('fs');
const app = express();

app.use(express.json());

app.post('/save', (req, res) => {
  const data = req.body;
  fs.writeFileSync('data.json', JSON.stringify(data));
  res.send('Data saved successfully');
});

app.get('/load', (req, res) => {
  if (fs.existsSync('data.json')) {
    const data = JSON.parse(fs.readFileSync('data.json'));
    res.json(data);
  } else {
    res.status(404).send('No data found');
  }
});

app.listen(3001, () => {
  console.log('Server is running on port 3001');
});