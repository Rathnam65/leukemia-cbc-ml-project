
# Machine Learning Based Leukemia Risk Evaluation Using CBC

This project implements a simple, end-to-end machine learning screening system
that uses routine Complete Blood Count (CBC) parameters to estimate leukemia risk
(Low / Medium / High).

## Tech Stack
- Python 3
- Flask (Backend API)
- scikit-learn (ML)
- HTML/CSS (Frontend)

## How to Run Locally
1. Install requirements:
   ```bash
   pip install -r backend/requirements.txt
   ```
2. Train the model (creates `backend/model/leukemia_model.pkl`):
   ```bash
   python backend/train_model.py
   ```
3. Run server:
   ```bash
   python backend/app.py
   ```
4. Open frontend/index.html in a browser.

## Deploying to a Cloud Platform (Docker)
This project is ready to run inside a container. Build and run with Docker:

```bash
# Build the image (from project root)
docker build -t leukemia-cbc-app .

# Run locally
docker run -p 5000:5000 leukemia-cbc-app
```

Then visit: `http://localhost:5000/`

### Cloud Deployment Notes
- This project includes a `Dockerfile` and `Procfile` for deployment on platforms like Azure App Service, AWS Elastic Beanstalk, Heroku, and similar.
- The server binds to `0.0.0.0` and reads `PORT` from the environment to support standard cloud hosting.
