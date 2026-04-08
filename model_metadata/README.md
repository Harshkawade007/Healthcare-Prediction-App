# Model Metadata

This folder tracks the deployment details for each saved model in `models/`.

Each JSON file records:
- saved model filename
- source notebook used to build it
- task / target meaning
- expected input fields for prediction
- any value mappings the API must apply
- whether feature engineering is required before inference
- basic performance metrics observed during packaging

Keep these files updated whenever a model is retrained or replaced.
