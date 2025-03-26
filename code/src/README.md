# AI-Driven Hyper-Personalization System

A prototype implementation of an AI-driven personalization and recommendation system. This system demonstrates key features of modern personalization including user profiling, content recommendations, and interaction tracking.

## Features

- User profile management with interest tagging
- Content-based recommendation engine using cosine similarity
- Real-time interaction tracking (likes, shares)
- Modern, responsive UI using Tailwind CSS
- RESTful API architecture

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python app.py
```

3. Access the web interface at `http://localhost:5000`

## System Components

- **User Profiling**: Users can define their interests which are used to create preference vectors
- **Content Recommendation**: Uses content-based filtering with cosine similarity to match user preferences with content
- **Interaction Tracking**: Records user interactions with content for future recommendation refinement
- **Modern UI**: Responsive interface built with Tailwind CSS for optimal user experience

## API Endpoints

- `/api/profile`: GET/POST user preferences
- `/api/recommend`: GET personalized recommendations
- `/api/interact`: POST user interactions with content

## Technology Stack

- Backend: Flask
- Database: SQLite with SQLAlchemy
- ML: scikit-learn
- Frontend: HTML5, JavaScript, Tailwind CSS
