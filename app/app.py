import atexit
import logging
import os
import uuid
from flask import Flask, g, session

from app.config import SECRET_KEY
from app.extensions import bcrypt, limiter
from app.persistence.storage import load_state, save_state, init_db, seed_default_user
from app.routes.alerts import alerts_bp
from app.routes.auth import auth_bp
from app.routes.dashboard import dashboard_bp
from app.routes.prediction import prediction_bp

# ---- Structured logging setup ----
from pythonjsonlogger import jsonlogger

def setup_logging() -> None:
    handler = logging.StreamHandler()
    formatter = jsonlogger.JsonFormatter(
        '%(asctime)s %(name)s %(levelname)s %(message)s %(request_id)s %(user)s'
    )
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)

setup_logging()
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    flask_app = Flask(__name__)
    flask_app.secret_key = SECRET_KEY

    # ---- Initialise extensions ----
    bcrypt.init_app(flask_app)
    limiter.init_app(flask_app)

    # Store bcrypt on app context for route access
    flask_app.bcrypt = bcrypt

    # ---- Register blueprints ----
    flask_app.register_blueprint(auth_bp)
    flask_app.register_blueprint(dashboard_bp)
    flask_app.register_blueprint(alerts_bp)
    flask_app.register_blueprint(prediction_bp)

    # ---- Request‑ID middleware (for logging) ----
    @flask_app.before_request
    def set_request_id():
        g.request_id = str(uuid.uuid4())[:8]

    # ---- Database initialisation & admin seeding ----
    with flask_app.app_context():
        init_db()
        seed_default_user(bcrypt)

    return flask_app


# Load state and register save on exit
load_state()
atexit.register(save_state)

app = create_app()


def main() -> None:
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug)


if __name__ == "__main__":
    main()