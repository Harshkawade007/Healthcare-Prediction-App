# AI-Driven Healthcare Portfolio

A multi-model healthcare prediction project with a FastAPI backend and a clean HTML/Tailwind frontend.

The platform supports 5 disease workflows:
1. Breast cancer risk classification
2. Diabetes risk classification
3. Heart disease risk classification
4. Liver cirrhosis stage classification
5. Lung disease image classification from chest X-rays

## What This Project Does

This project combines trained ML/DL models with a user-friendly UI and production-style API:
- Accepts user inputs from a frontend workspace
- Applies required preprocessing and feature engineering in the backend
- Returns prediction labels with probabilities/confidence
- Exposes model metadata to keep inference behavior transparent

## Tech Stack

- Backend: FastAPI, Uvicorn, Pydantic
- ML: CatBoost, XGBoost, scikit-learn pipelines
- DL (Lung): PyTorch + torchvision (EfficientNet-B0)
- Serialization: `joblib` (tabular models), `.pth` state dict (lung model)
- Frontend: HTML + Tailwind CSS + vanilla JavaScript

## Model Summary

| Disease | Model | Input Type | Key Notes | Performance (from project metadata) |
|---|---|---|---|---|
| Breast Cancer | `Pipeline(StandardScaler + CatBoostClassifier)` | Tabular | Feature-engineered lesion geometry + density interactions | Accuracy: 0.6647, Weighted F1: 0.6624 |
| Diabetes | `XGBClassifier` | Tabular | Encoded demographic + metabolic features | Accuracy: 0.9724, ROC-AUC: 0.9794 |
| Heart Disease | `CatBoostClassifier` | Tabular | Engineered interaction and burden features | Accuracy: 0.7109, ROC-AUC: 0.8371 |
| Liver Disease | `XGBClassifier` | Tabular | Multiclass cirrhosis stage model | Accuracy: 0.8979, Macro ROC-AUC: 0.9721 |
| Lung Disease | `EfficientNet-B0` | Image | 5-class chest X-ray classifier | Test Accuracy: 0.9180, Macro ROC-AUC: 0.9885 |

## Project Structure

```text
Dissertation/
  app/
    __init__.py
    app.py                 # FastAPI routes and inference logic
    schemas.py             # Request schemas
  models/
    breast_cancer.joblib
    diabetes.joblib
    heart_disease.joblib
    liver_disease.joblib
    lung_disease.pth
  model_metadata/
    breast_cancer.json
    diabetes.json
    heart_disease.json
    liver_disease.json
    lung_disease.json
  Prog/
    UI/
      page.html            # Frontend
    Project Files/         # Research notebooks/source experiments
  lung_disease_model.py    # Lung architecture + weight loader
  main.py                  # App launcher/entry point
  pyproject.toml
  requirements.txt
  uv.lock
```

## Local Setup

### 1. Create and activate virtual environment

```powershell
uv venv
.\.venv\Scripts\activate
```

### 2. Install dependencies

```powershell
uv sync
```

Alternative:

```powershell
pip install -r requirements.txt
```

## Run the Backend API

You can start the API in either of the following ways:

```powershell
uvicorn main:app --reload
```

or

```powershell
python main.py
```

API docs:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

## Run the Frontend

Open:
- `Prog/UI/page.html`

The frontend currently calls:
- `http://127.0.0.1:8000`

If you deploy the backend and frontend separately, update `API_BASE_URL` in `page.html` to your deployed backend URL.

## API Endpoints

### Health and metadata

- `GET /`
- `GET /health`
- `GET /metadata/{disease}`

`{disease}` can be:
- `breast_cancer`
- `diabetes`
- `heart_disease`
- `liver_disease`
- `lung_disease`

### Prediction endpoints

- `POST /predict/breast` (JSON)
- `POST /predict/diabetes` (JSON)
- `POST /predict/heart` (JSON)
- `POST /predict/liver` (JSON)
- `POST /predict/lung` (multipart image upload)

## Example Requests

### Diabetes prediction

```json
{
  "gender": "female",
  "age": 54,
  "hypertension": 0,
  "heart_disease": 0,
  "smoking_history": "no",
  "bmi": 27.3,
  "HbA1c_level": 6.6,
  "blood_glucose_level": 140
}
```

### Lung prediction (curl)

```bash
curl -X POST "http://127.0.0.1:8000/predict/lung" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/chest_xray.png"
```

## Output Format

Typical response fields:
- `disease`
- `predicted_class`
- `predicted_label`
- `probabilities`
- `positive_class_probability` (binary models)
- `class_labels`

The frontend shows a human-friendly summary by default and keeps raw request/response in collapsible technical details for debugging.

## Deployment Notes

Recommended flow:
1. Push this repo to GitHub
2. Deploy FastAPI backend (Render/Railway/Fly.io or similar)
3. Deploy frontend (Netlify/Vercel or static host)
4. Update `API_BASE_URL` in frontend
5. Restrict CORS in backend to your frontend domain

Important:
- If frontend is `https`, backend should also be `https`
- Keep model files with the deployment artifact

## Docker (Separate Backend and Frontend Images)

This project includes two Dockerfiles:
- `Dockerfile.backend` for FastAPI + models (uses `uv` for faster dependency installation and CPU-only PyTorch wheels)
- `Dockerfile.frontend` for static UI (Nginx)

### Build Backend Image

```bash
docker build -f Dockerfile.backend -t healthcare-backend:latest .
```

### Run Backend Container

```bash
docker run --rm -p 8000:8000 healthcare-backend:latest
```

Backend API will be available at:
- `http://127.0.0.1:8000`

### Build Frontend Image

```bash
docker build -f Dockerfile.frontend -t healthcare-frontend:latest .
```

### Run Frontend Container

```bash
docker run --rm -p 8080:80 healthcare-frontend:latest
```

Frontend will be available at:
- `http://127.0.0.1:8080`

Note:
- Before deploying frontend, ensure its `API_BASE_URL` points to your deployed backend URL.

### Run Both with Docker Compose

Use this for local full-stack startup:

```bash
docker compose up --build
```

Then open:
- Frontend: `http://127.0.0.1:8080`
- Backend docs: `http://127.0.0.1:8000/docs`

To stop:

```bash
docker compose down
```

## Important Disclaimer

This project is intended for educational/research support and software demonstration.
It is not a medical diagnosis system and must not replace professional clinical judgment.

## CI/CD (GitHub Actions)

This repo now includes:
- `.github/workflows/ci.yml`
- `.github/workflows/cd.yml`

### Pipeline Behavior

- CI runs on every push/PR to `main`:
  - Python syntax compile checks
  - `docker compose` config validation

- CD runs on push to `main` (or manual run):
  - Builds backend and frontend Docker images
  - Pushes images to Docker Hub (`latest` + commit SHA tag)
  - Triggers Render deploy hooks for backend and frontend

### Required GitHub Secrets

Go to:
`Repository -> Settings -> Secrets and variables -> Actions`

Add these secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
- `RENDER_BACKEND_DEPLOY_HOOK_URL`
- `RENDER_FRONTEND_DEPLOY_HOOK_URL`

If deploy hook secrets are missing, image push still works and deploy trigger steps are skipped.
