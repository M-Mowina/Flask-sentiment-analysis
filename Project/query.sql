-- Users Table
CREATE TABLE Users (
    user_id SECREATE TABLE Users (
    user_id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(100) NOT NULL,
    email VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE Feedback (
    feedback_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES Users(user_id),
    feedback_text TEXT NOT NULL,
    sentiment_score FLOAT, -- Optional
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE ContactUs (
    contact_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES Users(user_id), -- Optional
    name VARCHAR(100), -- Optional
    email VARCHAR(100) NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
EFAULT CURRENT_TIMESTAMP
);