# Free Online Deployment Guide

This project is a Flask application, so it cannot be deployed as a real working app on GitHub Pages. GitHub Pages only serves static files and cannot run Python, load the fraud model, handle login sessions, or serve prediction API endpoints.

Use Render's free web service instead.

## Live App Target

After deployment, the app URL will look like:

```text
https://credit-card-fraud-detector.onrender.com
```

The exact URL can vary if Render assigns a slightly different service slug.

## Files Added for Deployment

The repository includes:

```text
render.yaml
requirements.txt
```

`render.yaml` tells Render how to build and run the app:

```yaml
buildCommand: pip install -r requirements.txt
startCommand: gunicorn app:app
```

`requirements.txt` includes `gunicorn`, which is the production WSGI server used to run Flask online.

## Deployment Steps on Render

1. Go to:

```text
https://render.com
```

2. Sign in with GitHub.

3. Click `New +`.

4. Choose `Blueprint`.

5. Select this repository:

```text
salimshre/credit-card-fraud-detector
```

6. Render will detect:

```text
render.yaml
```

7. Confirm the service settings and create the blueprint.

8. Wait for the build and deploy process to finish.

## Render Settings

If creating the service manually instead of using Blueprint, use these values:

```text
Runtime: Python
Plan: Free
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

Environment variables:

```text
PYTHON_VERSION=3.11.9
APP_USERNAME=admin
APP_PASSWORD=admin123
SECRET_KEY=<generate a long random value>
FRAUD_STORE_PATH=/tmp/fraud-shield-data-store.json
```

For a public demo, change `APP_PASSWORD` in Render after the first deploy.

## Login

Default demo login:

```text
Username: admin
Password: admin123
```

Change these in Render's environment settings before sharing the site publicly.

## Important Free Plan Notes

Render's free service can sleep when inactive. The first request after sleeping may take some time to load.

This app stores runtime dashboard state in:

```text
/tmp/fraud-shield-data-store.json
```

On free hosting, this storage is temporary. For long-term production use, replace JSON storage with a database such as PostgreSQL.

## Verify Deployment

After Render finishes deploying, open the app URL and check:

```text
/health
```

Example:

```text
https://credit-card-fraud-detector.onrender.com/health
```

Expected result:

```text
status: ok
```

Then open the main dashboard and log in with the configured credentials.

## Redeploy in the Future

After the Render service is connected to GitHub, future deploys are simple:

```bash
git add .
git commit -m "Update fraud detector"
git push origin main
```

Render will automatically rebuild and redeploy the latest pushed code.

If your active branch is not `main`, either push the branch Render is watching or update the watched branch in Render's service settings.

## Troubleshooting

### Build Fails on Dependencies

Check the deploy logs in Render. Confirm that `requirements.txt` includes all required packages:

```text
Flask
pandas
numpy
scikit-learn
joblib
gunicorn
```

### App Starts but Prediction Fails

Confirm these files exist in the repository:

```text
fraud_model.pkl
scaler.pkl
model_metadata.json
```

The Flask app loads these files at runtime.

### Login Does Not Work

Check the Render environment variables:

```text
APP_USERNAME
APP_PASSWORD
SECRET_KEY
```

If they were changed after deployment, redeploy or restart the service.

### Runtime Data Disappears

This is expected on free ephemeral storage. Use PostgreSQL or another persistent database for production.
