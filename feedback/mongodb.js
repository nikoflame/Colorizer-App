import mongoose from 'mongoose';

const feedbackSchema = new mongoose.Schema({
  text: { type: String, required: true },
  created_at: { type: Date, default: Date.now },
});

// The 'Feedback' collection will be created in MongoDB
const Feedback = mongoose.model('Feedback', feedbackSchema);

export default Feedback;
