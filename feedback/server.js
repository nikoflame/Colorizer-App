import express from 'express';
import db from './db.js';
import cors from 'cors';

const app = express();

// Middleware to parse JSON bodies
app.use(express.json());

// CORS fix
app.use(cors({
  origin: 'http://localhost:10000'
}));

// POST /feedback endpoint to store feedback
app.post('/feedback', (req, res) => {
  const { feedback } = req.body;

  const sql = 'INSERT INTO feedback (text) VALUES (?)';
  db.run(sql, [feedback], function (err) {
    if (err) {
      console.error("Error inserting feedback:", err.message);
      return res.status(500).json({ error: 'Database error.' });
    }
    // this.lastID contains the ID of the inserted row
    res.json({ success: true, id: this.lastID });
  });
});

// GET /feedback endpoint to retrieve stored feedback
app.get('/feedback', (req, res) => {
  db.all('SELECT * FROM feedback ORDER BY created_at DESC', (err, rows) => {
    if (err) {
      console.error("Error fetching feedback:", err.message);
      return res.status(500).json({ error: 'Database error.' });
    }
    res.json(rows);
  });
});

const PORT = process.env.PORT || 10000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
