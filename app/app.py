import atexit
import logging

from flask import Flask

from app.config import SECRET_KEY
from app.persistence.storage import load_state, save_state
from app.routes.alerts import alerts_bp
from app.routes.auth import auth_bp
from app.routes.dashboard import dashboard_bp
from app.routes.prediction import prediction_bp

logging.basicConfig(level=logging.INFO)


def create_app() -> Flask:
    flask_app = Flask(__name__)
    flask_app.secret_key = SECRET_KEY

    flask_app.register_blueprint(auth_bp)
    flask_app.register_blueprint(dashboard_bp)
    flask_app.register_blueprint(alerts_bp)
    flask_app.register_blueprint(prediction_bp)

    return flask_app


load_state()
atexit.register(save_state)

app = create_app()


def main() -> None:
    import os

    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug)


if __name__ == "__main__":
    main()
