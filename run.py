"""
Face2Frame - AI Spectacle Recommendation System
Flask Web Application
"""

import os
import uuid
import logging
from pathlib import Path
from datetime import datetime

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, session
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "face2frame-dev-secret-key-change-in-production")

# Configuration
BASE_DIR = Path(__file__).parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Ensure upload folder exists
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# Model paths for PyTorch Hybrid model
MODEL_PATH = BASE_DIR / "models" / "face_shape_hybrid_final.pth"
LANDMARKER_PATH = BASE_DIR / "models" / "face_landmarker.task"
CATALOG_PATH = BASE_DIR / "data" / "glasses_metadata.xlsx"

# Global prediction service (initialized once, reused across requests)
_prediction_service = None


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_unique_filename(original_filename: str) -> str:
    """Generate unique filename to prevent overwrites."""
    ext = original_filename.rsplit(".", 1)[1].lower() if "." in original_filename else "jpg"
    unique_id = uuid.uuid4().hex[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{timestamp}_{unique_id}.{ext}"


def get_prediction_service():
    """
    Get or create the global prediction service.
    Initializes once and reuses across requests for better performance.
    """
    global _prediction_service
    
    if _prediction_service is None:
        from app.services.prediction_service import PredictionService
        
        _prediction_service = PredictionService(
            model_path=MODEL_PATH if MODEL_PATH.exists() else None,
            landmarker_path=LANDMARKER_PATH if LANDMARKER_PATH.exists() else None,
            catalog_path=CATALOG_PATH,
            enable_brightness_optimization=True,
        )
        logger.info("Prediction service initialized")
    
    return _prediction_service


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.route("/")
def index():
    """Landing page."""
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Handle image upload and AI prediction."""
    if request.method == "GET":
        return render_template("upload.html")
    
    # POST: Process uploaded image
    if "image" not in request.files:
        flash("No image file provided", "error")
        return redirect(request.url)
    
    file = request.files["image"]
    
    if file.filename == "":
        flash("No image selected", "error")
        return redirect(request.url)
    
    if not allowed_file(file.filename):
        flash(f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}", "error")
        return redirect(request.url)
    
    try:
        # Save uploaded file
        filename = generate_unique_filename(secure_filename(file.filename))
        filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
        file.save(str(filepath))
        logger.info(f"Image saved: {filepath}")
        
        # Run prediction (service is reused across requests)
        service = get_prediction_service()
        result = service.predict(str(filepath), top_n_recommendations=20)
        
        # Store recommendations in session for Load More API
        if result.success and result.recommendations:
            session["recommendations"] = result.recommendations
        else:
            session["recommendations"] = []
        
        # Prepare template data
        image_url = url_for("static", filename=f"uploads/{filename}")
        
        return render_template(
            "result.html",
            success=result.success,
            image_url=image_url,
            face_shape=result.face_shape,
            face_shape_confidence=result.face_shape_confidence,
            skin_tone=result.skin_tone,
            facial_metrics=result.facial_metrics,
            recommendations=result.recommendations,
            brightness_info=result.brightness_info,
            stability=result.stability,
            warnings=result.errors,
            error_message=result.error_message if not result.success else None,
        )
    
    except FileNotFoundError as e:
        logger.error("Catalog not found: %s", e)
        flash("Glasses catalog not found. Add data/glasses_metadata.xlsx.", "error")
        return redirect(request.url)
    except ValueError as e:
        logger.error("Config error: %s", e)
        flash(str(e), "error")
        return redirect(request.url)
    except Exception as e:
        logger.exception("Prediction failed")
        flash(f"An error occurred: {str(e)}", "error")
        return redirect(request.url)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """API endpoint for predictions (JSON response)."""
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400
    
    file = request.files["image"]
    
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file"}), 400
    
    try:
        filename = generate_unique_filename(secure_filename(file.filename))
        filepath = Path(app.config["UPLOAD_FOLDER"]) / filename
        file.save(str(filepath))
        
        service = get_prediction_service()
        result = service.predict(str(filepath), top_n_recommendations=20)
        if result.success and result.recommendations:
            session["recommendations"] = result.recommendations
        else:
            session["recommendations"] = []
        
        response = result.to_dict()
        response["image_url"] = url_for("static", filename=f"uploads/{filename}", _external=True)
        
        return jsonify(response)
    
    except Exception as e:
        logger.exception("API prediction failed")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/recommend", methods=["GET"])
def api_recommend():
    """Return a slice of stored recommendations (for Load More). Query: limit=10, offset=0."""
    limit = request.args.get("limit", 10, type=int)
    offset = request.args.get("offset", 0, type=int)
    limit = max(1, min(limit, 50))
    offset = max(0, offset)
    recs = session.get("recommendations", [])
    chunk = recs[offset : offset + limit]
    return jsonify({
        "success": True,
        "recommendations": chunk,
        "offset": offset,
        "limit": limit,
        "total": len(recs),
        "has_more": offset + len(chunk) < len(recs),
    })


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "hybrid_model_available": MODEL_PATH.exists(),
        "landmarker_available": LANDMARKER_PATH.exists(),
        "catalog_available": CATALOG_PATH.exists(),
    })


# -----------------------------------------------------------------------------
# Error Handlers
# -----------------------------------------------------------------------------

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    flash("File is too large. Maximum size is 16MB.", "error")
    return redirect(url_for("predict"))


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template("error.html", error="Page not found", code=404), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    return render_template("error.html", error="Internal server error", code=500), 500


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
