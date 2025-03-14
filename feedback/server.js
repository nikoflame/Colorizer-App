import express from 'express';
import db from './db.js';
import Feedback from './mongodb.js';
import cors from 'cors';
import mongoose from 'mongoose';

const app = express();
const MONGODB_URI = process.env.MONGODB_URI;

// Middleware to parse JSON bodies
app.use(express.json());

// CORS fix
app.use(cors({
  origin: 'https://colorizer-app.onrender.com'
}));


// MongoDB connection
mongoose.connect(MONGODB_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
  .then(() => console.log("Connected to MongoDB!"))
  .catch((err) => console.error("MongoDB connection error:", err));

// POST /feedback - store feedback in MongoDB
app.post('/feedback', async (req, res) => {
  try {
    const { feedback } = req.body;
    // Insert a new document
    const newFeedback = await Feedback.create({ text: feedback });
    // Return success + the newly created ID
    return res.json({ success: true, id: newFeedback._id });
  } catch (err) {
    console.error("Error inserting feedback:", err);
    return res.status(500).json({ error: 'Database error.' });
  }
});

// GET /feedback - retrieve stored feedback from MongoDB
app.get('/feedback', async (req, res) => {
  try {
    // Find all feedback, sort by creation date desc
    const allFeedback = await Feedback.find().sort({ created_at: -1 });
    return res.json(allFeedback);
  } catch (err) {
    console.error("Error fetching feedback:", err);
    return res.status(500).json({ error: 'Database error.' });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}`);
});
