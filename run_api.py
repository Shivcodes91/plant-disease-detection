# run_api.py
# Simple script to start the API
# Run with: python run_api.py

import uvicorn

if __name__ == "__main__":
    print("Starting Plant Disease Detection API...")
    print("API docs available at: http://localhost:8000/docs")
    print("Press Ctrl+C to stop\n")

    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,      # auto-restart when code changes
        reload_dirs=["backend", "ml_training", "data"]
    )