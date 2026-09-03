from fastapi import FastAPI

app = FastAPI(
    title="Piyush Portfolio Demo API",
    description="Backend API for portfolio project demonstrations",
    version="1.0.0"
)


@app.get("/")
def home():
    return {
        "message": "Piyush's Portfolio API is running!"
    }


@app.get("/api/about")
def about():
    return {
        "name": "Piyush Mhatre",
        "role": "Backend Developer",
        "language": "Python",
        "framework": "FastAPI"
    }


@app.get("/api/project")
def project():
    return {
        "name": "Explainable AI Financial Advisor",
        "status": "Currently rebuilding",
        "backend": "FastAPI",
        "frontend": "React",
        "type": "AI-powered financial application"
    }