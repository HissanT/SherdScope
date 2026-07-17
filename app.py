"""SherdScope Flask application and compatibility API routes."""

from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from pathlib import Path
import os
import re
import json
import base64
from io import BytesIO
import pandas as pd
import numpy as np
from werkzeug.utils import secure_filename
import torch
import hashlib
import threading
import shutil
import tempfile
import secrets

from utils import (
    PDFProcessor,
    ModelProcessor,
    MaskExtractor,
    AnnotationProcessor,
    ImageProcessor,
    TabularProcessor,
    SecondStepProcessor,
    ExportProcessor,
    PDFConfig,
    ModelConfig,
    MaskExtractionConfig,
    AnnotationConfig,
    TabularConfig,
    SecondStepConfig,
    ExportConfig,
    read_vessels_sidecar,
    write_vessels_sidecar,
    VESSELS_SIDECAR_SUFFIX,
    read_scale_sidecar,
    write_scale_sidecar,
)

from project_manager import ProjectManager
from metadata_linker import (
    HESBAN_TABLE_COLUMNS,
    WARNING_OVERRIDE_REASONS,
    Hesban11Profile,
    MetadataLinkError,
    ReviewerRevisionConflict,
    MetadataLinker,
    StructuredExtractor,
    apply_approved_figures,
    ensure_page_manifest,
    get_profile,
    linkage_totals,
    load_linkage_state,
    natural_key,
    normalize_figure_id,
    migrate_linkage_columns,
    record_pdf_pages,
    save_linkage_state,
    validate_figure,
)
from ocr_extractor import PaddleOCRStructuredExtractor
from hesban_measurements import (
    manual_calibration,
    measure_figure,
    persist_figure_measurements,
    verified_measurement,
)
from routes.research_export import register_research_export_routes

# Stable aliases used by the shared JSON-repair helpers below.
_re = re
_json = json

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SHERDSCOPE_SECRET_KEY') or secrets.token_hex(32)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['UPLOAD_FOLDER'] = Path('temp_uploads')
app.config['UPLOAD_FOLDER'].mkdir(exist_ok=True)

# Initialize Project Manager
project_manager = ProjectManager(projects_root="projects")
register_research_export_routes(app, lambda: project_manager)

# === Gemma 4 AI model (lazy loaded for bibliographic extraction) ===
_gemma_model = None
_gemma_processor = None
_gemma_model_lock = threading.Lock()

def load_gemma_model(hf_token=None):
    """Lazy-load google/gemma-4-E2B-it for vision-based bibliographic extraction."""
    global _gemma_model, _gemma_processor
    with _gemma_model_lock:
        if _gemma_model is None:
            from transformers import AutoProcessor, AutoModelForMultimodalLM
            model_id = "google/gemma-4-E2B-it"
            token = hf_token or os.environ.get('HF_TOKEN', '') or None
            cache_dir = Path("models_llm")
            cache_dir.mkdir(exist_ok=True)
            print(f"[AI] Loading {model_id} into {cache_dir}...")

            # Signal download progress to frontend (indeterminate — HF Hub manages the actual download)
            operation_progress['active'] = True
            operation_progress['operation'] = 'model_download'
            operation_progress['message'] = 'Downloading Gemma 4 E2B model (~10 GB) — this may take a while...'
            operation_progress['percent'] = 0
            operation_progress['current'] = 0
            operation_progress['total'] = 1

            _gemma_processor = AutoProcessor.from_pretrained(
                model_id, token=token, cache_dir=cache_dir
            )

            operation_progress['message'] = 'Loading model weights into GPU memory...'
            operation_progress['percent'] = 70

            if torch.cuda.is_available():
                device_map = "auto"
            else:
                # MPS (Apple Silicon) has a max single-buffer limit (~4GB) that
                # prevents loading Gemma 4 E2B (5.1B params). Use CPU instead.
                device_map = {"": "cpu"}
            _gemma_model = AutoModelForMultimodalLM.from_pretrained(
                model_id,
                token=token,
                dtype="auto",
                device_map=device_map,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True
            )
            operation_progress['message'] = 'Model ready'
            operation_progress['percent'] = 100
            operation_progress['active'] = False
            print(f"[AI] Gemma 4 E2B-it loaded (device_map={device_map})")
    return _gemma_model, _gemma_processor


def _annotate_image_with_bboxes(image, bbox_data):
    """Draw labeled bounding boxes on a PIL RGB image before sending to VLM.

    bbox_data: list of (label, x1, y1, x2, y2) in image pixel coordinates.
    Draws an orange rectangle for each box and a small filled label at its
    top-left corner so the model can visually locate each drawing by ID.
    Returns the mutated (in-place) PIL image.
    """
    from PIL import ImageDraw, ImageFont

    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    # Scale strokes and font to image size
    line_width = max(2, int(min(img_w, img_h) / 350))
    font_size  = max(14, int(min(img_w, img_h) / 55))

    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            font = ImageFont.truetype(font_path, font_size)
            break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()

    BOX_COLOR = (255, 80, 0)    # vivid orange — visible on both light and dark bg
    LABEL_BG  = (255, 80, 0)
    LABEL_FG  = (255, 255, 255)
    pad = max(3, line_width)

    for label, x1, y1, x2, y2 in bbox_data:
        # Bounding box rectangle
        draw.rectangle([x1, y1, x2, y2], outline=BOX_COLOR, width=line_width)

        # Measure label text
        try:
            bb = font.getbbox(str(label))
            tw, th = bb[2] - bb[0], bb[3] - bb[1]
        except Exception:
            tw, th = font_size * len(str(label)), font_size

        # Place label above the box; fall back to inside-top if it would go off-screen
        lx = x1
        ly = y1 - th - pad * 2
        if ly < 0:
            ly = y1 + pad

        draw.rectangle([lx, ly, lx + tw + pad * 2, ly + th + pad * 2], fill=LABEL_BG)
        draw.text((lx + pad, ly + pad), str(label), fill=LABEL_FG, font=font)

    return image


class VisionUnsupportedError(Exception):
    """Raised when the selected OpenRouter model does not support image/vision input."""
    pass


def call_openrouter_ai(image_pil, prompt, api_key, model_name, max_tokens=2048):
    """Send an image + text prompt to OpenRouter and return the raw text response.

    The image is base64-encoded and sent as an OpenAI-compatible vision message so
    any vision-capable model available on OpenRouter can be used. ``max_tokens`` is
    sized by the caller to the number of drawings so busy pages are not truncated.
    """
    import base64 as _base64
    import io as _io
    from openai import OpenAI

    images = image_pil if isinstance(image_pil, (list, tuple)) else [image_pil]
    content = []
    for image in images:
        buf = _io.BytesIO()
        image.save(buf, format='PNG')
        img_b64 = _base64.b64encode(buf.getvalue()).decode('utf-8')
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/png;base64,{img_b64}"}})
    content.append({"type": "text", "text": prompt})

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
            max_tokens=max_tokens,
        )
    except Exception as _or_exc:
        _msg = str(_or_exc)
        if '404' in _msg or 'image input' in _msg.lower() or 'No endpoints' in _msg:
            raise VisionUnsupportedError(
                f"Model '{model_name}' does not support image/vision input on OpenRouter. "
                "Please select a vision-capable model in the AI Backend panel."
            ) from _or_exc
        raise

    return response.choices[0].message.content or ""


init_status = {
    'ready': False,
    'stage': 'starting',
    'progress': 0,
    'message': 'Initializing...'
}

def update_init_status(stage, progress, message):
    """Update initialization status"""
    global init_status
    init_status['stage'] = stage
    init_status['progress'] = progress
    init_status['message'] = message
    print(f"[Init] {progress}% - {message}")

# Global progress tracking
operation_progress = {
    'active': False,
    'operation': '',
    'total': 0,
    'current': 0,
    'message': '',
    'percent': 0
}

def update_operation_progress(operation, current, total, message=''):
    """Update operation progress for frontend polling"""
    global operation_progress
    operation_progress['active'] = True
    operation_progress['operation'] = operation
    operation_progress['current'] = current
    operation_progress['total'] = total
    operation_progress['message'] = message
    operation_progress['percent'] = int((current / total * 100)) if total > 0 else 0
    print(f"[{operation}] {operation_progress['percent']}% - {message}")

def clear_operation_progress():
    """Clear operation progress"""
    global operation_progress
    operation_progress['active'] = False
    operation_progress['operation'] = ''
    operation_progress['total'] = 0
    operation_progress['current'] = 0
    operation_progress['message'] = ''
    operation_progress['percent'] = 0

# Initialize directories
ROOT_DIR = Path(".")
PRED_OUTPUT_DIR = ROOT_DIR / "outputs"
PDFIMG_OUTPUT_DIR = ROOT_DIR / "pdf2img_outputs"
MODELS_DIR = ROOT_DIR / "models_vision"
MODELS_CLASSIFIER_DIR = ROOT_DIR / "models_classifier"
ASSETS_DIR = ROOT_DIR / "imgs"

# Create necessary directories
for directory in [PDFIMG_OUTPUT_DIR, MODELS_DIR, PRED_OUTPUT_DIR, MODELS_CLASSIFIER_DIR]:
    directory.mkdir(exist_ok=True)


# ==================== MODEL INITIALIZATION ====================

def download_model(url, destination, model_name, base_progress, progress_range):
    """Download a model file with progress tracking for splash screen"""
    import urllib.request
    import sys
    
    print(f"\n📥 Downloading {model_name}...")
    print(f"   URL: {url}")
    print(f"   Destination: {destination}")
    
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(downloaded * 100 / total_size, 100)
            
            # Update splash screen progress
            current_progress = base_progress + (percent / 100.0) * progress_range
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            update_init_status('downloading_models', current_progress, 
                             f'Downloading {model_name}: {downloaded_mb:.1f}/{total_mb:.1f} MB')
            
            # Console progress bar
            bar_length = 50
            filled_length = int(bar_length * percent / 100)
            bar = '█' * filled_length + '░' * (bar_length - filled_length)
            
            sys.stdout.write(f'\r   [{bar}] {percent:.1f}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)')
            sys.stdout.flush()
    
    try:
        urllib.request.urlretrieve(url, destination, show_progress)
        print(f"\n   ✅ Successfully downloaded {model_name}")
        return True
    except Exception as e:
        print(f"\n   ❌ Error downloading {model_name}: {e}")
        return False


def initialize_models():
    """Check and download required models if missing"""
    update_init_status('checking_models', 10, 'Checking required models...')
    
    print("\n" + "="*80)
    print(" 🔍 Checking Required Models")
    print("="*80)
    
    models_to_check = {
        'Vision Model': {
            'path': MODELS_DIR / 'BasicModelv8_v01.pt',
            'url': 'https://huggingface.co/lrncrd/PyPotteryLens/resolve/main/BasicModelv8_v01.pt'
        },
        'Classifier Model': {
            'path': MODELS_CLASSIFIER_DIR / 'model_classifier.pth',
            'url': 'https://huggingface.co/lrncrd/PyPotteryLens/resolve/main/model_classifier.pth'
        }
    }
    
    missing_models = []
    
    for model_name, info in models_to_check.items():
        if info['path'].exists():
            print(f"✅ {model_name}: Found at {info['path']}")
        else:
            print(f"❌ {model_name}: Not found at {info['path']}")
            missing_models.append((model_name, info))
    
    if missing_models:
        print(f"\n⚠️  {len(missing_models)} model(s) need to be downloaded")
        print("="*80)
        
        progress_per_model = 60 / len(missing_models)  # 60% total for models (20% -> 80%)
        
        for idx, (model_name, info) in enumerate(missing_models):
            base_progress = 20 + (idx * progress_per_model)
            success = download_model(info['url'], info['path'], model_name, 
                                   base_progress, progress_per_model)
            
            if not success:
                print(f"\n⚠️  Warning: Could not download {model_name}")
                print(f"   Please download manually from: {info['url']}")
                print(f"   And place it at: {info['path']}")
        
        print("\n" + "="*80)
        print(" ✨ Model initialization complete!")
        print("="*80)
    else:
        print("\n✨ All required models are present!")
        print("="*80)
    
    update_init_status('models_ready', 80, 'Models ready, initializing processors...')


def initialize_processors():
    """Initialize all processors after models are ready"""
    global pdf_processor, model_processor, mask_extractor, annotation_processor
    global image_processor, tabular_processor, second_step_processor, export_processor
    
    update_init_status('init_processors', 85, 'Initializing PDF processor...')
    pdf_processor = PDFProcessor(PDFConfig(output_dir=PDFIMG_OUTPUT_DIR))

    update_init_status('init_processors', 88, 'Initializing model processor...')
    model_processor = ModelProcessor(ModelConfig(
        models_dir=MODELS_DIR,
        pred_output_dir=PRED_OUTPUT_DIR
    ))

    update_init_status('init_processors', 90, 'Initializing mask extractor...')
    mask_extractor = MaskExtractor(MaskExtractionConfig(
        pdfimg_output_dir=PDFIMG_OUTPUT_DIR,
        pred_output_dir=PRED_OUTPUT_DIR
    ))

    update_init_status('init_processors', 92, 'Initializing annotation processor...')
    annotation_processor = AnnotationProcessor(AnnotationConfig(
        pred_output_dir=PRED_OUTPUT_DIR
    ))

    update_init_status('init_processors', 94, 'Initializing image processor...')
    image_processor = ImageProcessor(
        pdfimg_output_dir=PDFIMG_OUTPUT_DIR,
        pred_output_dir=PRED_OUTPUT_DIR
    )

    update_init_status('init_processors', 96, 'Initializing tabular processor...')
    tabular_processor = TabularProcessor(TabularConfig(
        pdfimg_output_dir=PDFIMG_OUTPUT_DIR,
        pred_output_dir=PRED_OUTPUT_DIR
    ))

    update_init_status('init_processors', 98, 'Initializing classification processor...')
    second_step_processor = SecondStepProcessor(SecondStepConfig(
        pred_output_dir=PRED_OUTPUT_DIR,
        model_path=MODELS_CLASSIFIER_DIR / "model_classifier.pth"
    ))

    update_init_status('init_processors', 99, 'Initializing export processor...')
    export_processor = ExportProcessor(ExportConfig(
        pred_output_dir=PRED_OUTPUT_DIR
    ))

    # Mark as ready
    update_init_status('ready', 100, 'Application ready!')
    init_status['ready'] = True
    print("\n" + "="*80)
    print(" ✅ All processors initialized - Application ready!")
    print("="*80)


# Initialize processor variables as None
pdf_processor = None
model_processor = None
mask_extractor = None
annotation_processor = None
image_processor = None
tabular_processor = None
second_step_processor = None
export_processor = None


def background_initialization():
    """Run initialization in background thread"""
    import threading
    
    def init_thread():
        try:
            update_init_status('init_start', 5, 'Starting initialization...')
            initialize_models()
            initialize_processors()
        except Exception as e:
            print(f"ERROR during initialization: {e}")
            import traceback
            traceback.print_exc()
            update_init_status('error', 0, f'Initialization failed: {e}')
    
    thread = threading.Thread(target=init_thread, daemon=True)
    thread.start()
    print("🚀 Background initialization started...")


# Start background initialization
if os.environ.get('PYPOTTERYLENS_SKIP_INIT') != '1':
    background_initialization()
else:
    init_status.update({'ready': True, 'stage': 'ready', 'progress': 100,
                        'message': 'Initialization skipped for testing'})


# ==================== ROUTES ====================

@app.route('/api/init-status')
def get_init_status():
    """Get initialization status"""
    return jsonify(init_status)

@app.route('/api/operation-progress')
@app.route('/api/progress')  # alias kept for backwards compatibility
def get_operation_progress():
    """Get current operation progress for frontend polling"""
    return jsonify(operation_progress)

@app.route('/api/system-info')
def get_system_info():
    """Get system information including CPU, GPU, and MPS availability"""
    try:
        import os
        import torch
        
        system_info = {
            'cpu': {
                'cores': os.cpu_count() or 1,
                'available': True
            },
            'gpu': {
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
            },
            'mps': {
                'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
            }
        }
        
        return jsonify(system_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/check-ai-requirements')
def check_ai_requirements():
    """Check if the system meets requirements for AI bibliographic extraction:
    - CUDA GPU with at least 6 GB VRAM
    - Whether the Gemma model is already cached locally
    """
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        vram_gb = 0.0
        gpu_name = ''
        if cuda_available and torch.cuda.device_count() > 0:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            gpu_name = props.name

        # Check if model blobs exist in the local cache directory
        model_cache_dir = Path("models_llm") / "models--google--gemma-4-E2B-it"
        model_cached = model_cache_dir.exists() and any(model_cache_dir.rglob("*.safetensors"))

        meets_requirements = cuda_available and vram_gb >= 6.0

        return jsonify({
            'cuda_available': cuda_available,
            'vram_gb': round(vram_gb, 2),
            'gpu_name': gpu_name,
            'model_cached': model_cached,
            'meets_requirements': meets_requirements
        })
    except Exception as e:
        return jsonify({'error': str(e), 'cuda_available': False,
                        'vram_gb': 0, 'gpu_name': '', 'model_cached': False,
                        'meets_requirements': False}), 500


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

# ============================================================================
# PROJECT MANAGEMENT API ROUTES
# ============================================================================

@app.route('/api/projects', methods=['GET'])
def list_projects():
    """Get list of all projects"""
    try:
        projects = project_manager.list_projects()
        return jsonify({'projects': projects, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects', methods=['POST'])
def create_project():
    """Create a new project"""
    try:
        data = request.get_json()
        project_name = data.get('project_name', '').strip()
        description = data.get('description', '').strip()
        icon = data.get('icon', '1.png')
        
        if not project_name:
            return jsonify({'error': 'Project name is required', 'success': False}), 400
        
        metadata = project_manager.create_project(project_name, description, icon)
        return jsonify({'project': metadata, 'success': True})
    except ValueError as e:
        return jsonify({'error': str(e), 'success': False}), 400
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get project metadata"""
    try:
        metadata = project_manager.get_project(project_id)
        if metadata is None:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        return jsonify({'project': metadata, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project"""
    try:
        success = project_manager.delete_project(project_id)
        if not success:
            return jsonify({'error': 'Project not found or could not be deleted', 'success': False}), 404
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/icons', methods=['GET'])
def get_icons():
    """Get list of available project icons"""
    try:
        icons_path = Path('static/imgs/icons')
        if not icons_path.exists():
            return jsonify({'icons': [], 'success': True})

        # Get all PNG files in static/imgs/icons folder
        icons = [f.name for f in icons_path.iterdir()
                if f.is_file() and f.suffix.lower() == '.png' and f.name != 'LogoLens.png']
        
        # Sort icons numerically if they are numbered
        try:
            icons.sort(key=lambda x: int(x.replace('.png', '')))
        except Exception:
            icons.sort()
        
        return jsonify({'icons': icons, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/icons/<filename>')
def serve_icon(filename):
    """Serve an icon file"""
    try:
        icons_path = Path('static/imgs/icons')
        return send_from_directory(icons_path, filename)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 404


@app.route('/api/projects/<project_id>/workflow', methods=['PATCH'])
def update_workflow_status(project_id):
    """Update workflow status for a project"""
    try:
        data = request.get_json()
        status_updates = data.get('status_updates', {})
        
        success = project_manager.update_workflow_status(project_id, status_updates)
        if not success:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Return updated metadata
        metadata = project_manager.get_project(project_id)
        return jsonify({'project': metadata, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/settings', methods=['PATCH'])
def update_project_settings(project_id):
    """Update project settings"""
    try:
        data = request.get_json()
        settings = data.get('settings', {})
        
        success = project_manager.update_settings(project_id, settings)
        if not success:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Return updated metadata
        metadata = project_manager.get_project(project_id)
        return jsonify({'project': metadata, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/excluded-images', methods=['POST'])
def update_excluded_images(project_id):
    """Update excluded images list for a project"""
    try:
        data = request.get_json()
        excluded_images = data.get('excluded_images', [])
        
        success = project_manager.update_excluded_images(project_id, excluded_images)
        if not success:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/reviewed', methods=['POST'])
def add_reviewed_image(project_id):
    """Mark an image as reviewed"""
    try:
        data = request.get_json()
        image_name = data.get('image_name', '')
        
        if not image_name:
            return jsonify({'error': 'Image name is required', 'success': False}), 400
        
        success = project_manager.add_reviewed_image(project_id, image_name)
        if not success:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/images', methods=['GET'])
def get_project_images(project_id):
    """Get list of images in project"""
    try:
        images = project_manager.get_images_list(project_id, 'images')
        
        if images is None:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Create URLs for each image
        image_urls = [f'/api/projects/{project_id}/image/{img}' for img in images]
        
        return jsonify({
            'images': image_urls,
            'count': len(images),
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/image/<filename>', methods=['GET'])
def serve_project_image(project_id, filename):
    """Serve an image from project's images folder"""
    try:
        images_path = project_manager.get_project_path(project_id, 'images')
        
        if not images_path or not images_path.exists():
            return jsonify({'error': 'Project or images folder not found', 'success': False}), 404
        
        return send_from_directory(images_path, filename)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 404


@app.route('/api/projects/<project_id>/masks', methods=['GET'])
def get_project_masks(project_id):
    """Get list of mask images in project"""
    try:
        masks = project_manager.get_images_list(project_id, 'masks')
        
        if masks is None:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Create URLs for each mask
        mask_urls = [f'/api/projects/{project_id}/mask/{img}' for img in masks]
        
        return jsonify({
            'masks': mask_urls,
            'count': len(masks),
            'success': True
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/mask/<filename>', methods=['GET'])
def serve_project_mask(project_id, filename):
    """Serve a mask image from project's masks folder"""
    try:
        masks_path = project_manager.get_project_path(project_id, 'masks')
        
        if not masks_path or not masks_path.exists():
            return jsonify({'error': 'Project or masks folder not found', 'success': False}), 404
        
        return send_from_directory(masks_path, filename)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 404


@app.route('/api/projects/<project_id>/masks/extract', methods=['POST'])
def extract_project_masks(project_id):
    """Extract cards from masks in project with progress tracking"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Get project paths
        masks_path = project_manager.get_project_path(project_id, 'masks')
        cards_path = project_manager.get_project_path(project_id, 'cards')
        
        if not masks_path or not masks_path.exists():
            return jsonify({'error': 'Project masks folder not found', 'success': False}), 404
        
        # Count mask files for progress
        mask_files = [f for f in masks_path.iterdir() if f.name.endswith('_mask_layer.png')]
        total_masks = len(mask_files)
        
        if total_masks == 0:
            return jsonify({'error': 'No mask files found. Apply a model first.', 'success': False}), 404
        
        # Initialize progress
        update_operation_progress('extract_masks', 0, total_masks, 'Starting extraction...')
        
        # Run extraction in a way that allows progress updates
        # We'll monkey-patch the print function temporarily
        import builtins
        original_print = builtins.print
        
        def progress_print(*args, **kwargs):
            msg = ' '.join(str(arg) for arg in args)
            # Look for progress pattern "Processing mask X/Y"
            if 'Processing mask' in msg:
                try:
                    parts = msg.split()
                    idx = parts.index('mask') + 1
                    current = int(parts[idx].split('/')[0])
                    update_operation_progress('extract_masks', current, total_masks, 
                                             f'Extracting mask {current}/{total_masks}')
                except Exception:
                    pass
            original_print(*args, **kwargs)
        
        builtins.print = progress_print
        
        try:
            # Extract masks using project paths
            result = mask_extractor.extract_masks_from_project(
                str(masks_path),
                str(cards_path)
            )
        finally:
            builtins.print = original_print
            clear_operation_progress()
        
        # Update project workflow status
        card_count = len(list(cards_path.glob('*.png'))) if cards_path.exists() else 0
        project_manager.update_workflow_status(project_id, {
            'cards_extracted': card_count
        })
        
        return jsonify({
            'message': result,
            'success': True
        })
        
    except Exception as e:
        clear_operation_progress()
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/masks/save', methods=['POST'])
def save_project_mask(project_id):
    """Save edited mask for a project image"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Get uploaded mask file
        if 'mask' not in request.files:
            return jsonify({'error': 'No mask file provided', 'success': False}), 400
        
        mask_file = request.files['mask']
        if mask_file.filename == '':
            return jsonify({'error': 'Empty filename', 'success': False}), 400
        
        # Save to project masks folder
        masks_path = project_manager.get_project_path(project_id, 'masks')
        filename = secure_filename(mask_file.filename)
        mask_filepath = masks_path / filename
        
        mask_file.save(mask_filepath)
        
        # Generate mask URL for frontend
        mask_url = f'/api/projects/{project_id}/mask/{filename}'
        
        return jsonify({
            'message': f'Mask saved: {filename}',
            'filename': filename,
            'mask_url': mask_url,
            'success': True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# ============================================================================
# MANUALLY DRAWN VESSELS (LabelMe-style polygons)
# During review the operator draws polygons around vessels the model missed
# (e.g. a vessel drawn inside another). Each polygon becomes its own card on
# extraction. Polygons are stored as a JSON sidecar next to the mask.
# ============================================================================

@app.route('/api/projects/<project_id>/vessels/<base>', methods=['GET'])
def get_image_vessels(project_id, base):
    """Return manually drawn vessel polygons for a single image."""
    try:
        masks_path = project_manager.get_project_path(project_id, 'masks')
        if not masks_path:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        return jsonify({'success': True, 'polygons': read_vessels_sidecar(masks_path, base)})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/vessels/<base>', methods=['POST'])
def save_image_vessels(project_id, base):
    """Persist manually drawn vessel polygons for a single image."""
    try:
        masks_path = project_manager.get_project_path(project_id, 'masks')
        if not masks_path or not masks_path.exists():
            return jsonify({'error': 'Project masks folder not found', 'success': False}), 404
        data = request.get_json(silent=True) or {}
        polygons = data.get('polygons', [])
        write_vessels_sidecar(masks_path, base, polygons)
        return jsonify({'success': True, 'count': len(polygons)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/vessels-summary', methods=['GET'])
def get_project_vessels_summary(project_id):
    """Per-image count of manually drawn vessels (for badges in the image list)."""
    try:
        masks_path = project_manager.get_project_path(project_id, 'masks')
        summary = {}
        if masks_path and masks_path.exists():
            for f in masks_path.iterdir():
                if not f.name.endswith(VESSELS_SIDECAR_SUFFIX):
                    continue
                base = f.name[:-len(VESSELS_SIDECAR_SUFFIX)]
                count = len(read_vessels_sidecar(masks_path, base))
                if count:
                    summary[base] = count
        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/scale/<base>', methods=['GET'])
def get_image_scale(project_id, base):
    """Return scale calibration entries for a single image."""
    try:
        masks_path = project_manager.get_project_path(project_id, 'masks')
        if not masks_path:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        return jsonify({'success': True, 'scales': read_scale_sidecar(masks_path, base)})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/scale/<base>', methods=['POST'])
def save_image_scale(project_id, base):
    """Persist scale calibration entries for a single image."""
    try:
        masks_path = project_manager.get_project_path(project_id, 'masks')
        if not masks_path or not masks_path.exists():
            return jsonify({'error': 'Project masks folder not found', 'success': False}), 404
        data = request.get_json(silent=True) or {}
        scales = data.get('scales', [])
        write_scale_sidecar(masks_path, base, scales)
        return jsonify({'success': True, 'count': len(scales)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# ============================================================================
# LEGACY API ROUTES (will be refactored to use projects)
# ============================================================================

@app.route('/api/folders/images')
def get_image_folders():
    """Get list of image folders"""
    try:
        folders = [f for f in os.listdir(PDFIMG_OUTPUT_DIR) 
                  if os.path.isdir(PDFIMG_OUTPUT_DIR / f)]
        return jsonify({'folders': folders, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/folders/masks')
def get_mask_folders():
    """Get list of folders with masks (for annotation tab)"""
    try:
        # Get folders ending with _mask
        folders = [f.replace('_mask', '') for f in os.listdir(PRED_OUTPUT_DIR) 
                  if f.endswith('_mask') and os.path.isdir(PRED_OUTPUT_DIR / f)]
        return jsonify({'folders': folders, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/folders/results')
def get_results_folders():
    """Get list of result folders"""
    try:
        folders = [f for f in os.listdir(PRED_OUTPUT_DIR) 
                  if f.endswith('_card') and os.path.isdir(PRED_OUTPUT_DIR / f)]
        return jsonify({'folders': folders, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/models')
def get_models():
    """Get list of available models"""
    try:
        models = [f for f in os.listdir(MODELS_DIR) 
                 if f.endswith('.pt')]
        return jsonify({'models': models, 'success': True})
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== PDF PROCESSING ====================

@app.route('/api/pdf/upload', methods=['POST'])
def upload_pdf():
    """Upload and process PDF file into a project"""
    try:
        print("PDF upload request received")
        print("Files:", request.files)
        print("Form data:", request.form)
        
        if 'file' not in request.files:
            print("Error: No file in request")
            return jsonify({'error': 'No file provided', 'success': False}), 400
        
        file = request.files['file']
        split_pages = request.form.get('split_pages', 'false').lower() == 'true'
        try:
            render_dpi = int(request.form.get('render_dpi', '400'))
        except (TypeError, ValueError):
            return jsonify({'error': 'Render DPI must be a whole number', 'success': False}), 400
        if not 200 <= render_dpi <= 600:
            return jsonify({'error': 'Render DPI must be between 200 and 600', 'success': False}), 400
        project_id = request.form.get('project_id', '').strip()
        
        print(f"File name: {file.filename}")
        print(f"Split pages: {split_pages}")
        print(f"Render DPI: {render_dpi}")
        print(f"Project ID: {project_id}")
        
        if not project_id:
            return jsonify({'error': 'No project selected', 'success': False}), 400
        
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        if file.filename == '':
            print("Error: Empty filename")
            return jsonify({'error': 'No file selected', 'success': False}), 400
        
        if not file.filename.lower().endswith('.pdf'):
            print("Error: Not a PDF file")
            return jsonify({'error': 'Only PDF files are allowed', 'success': False}), 400
        
        # Save PDF to project's pdf_source folder
        filename = secure_filename(file.filename)
        if not filename:
            return jsonify({'error': 'PDF filename is invalid', 'success': False}), 400
        pdf_source_path = project_manager.get_project_path(project_id, 'pdf_source')
        pdf_filepath = pdf_source_path / filename
        existing_sources = [p for p in pdf_source_path.glob('*.pdf') if p.name != filename]
        manifest_path = project_manager.get_project_path(project_id) / 'page_manifest.json'
        existing_manifest = {}
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as handle:
                existing_manifest = json.load(handle)
        old_source_pages = [page for page in existing_manifest.get('pages', [])
                            if page.get('source_pdf') == filename]
        if old_source_pages:
            previous_dpi = int(old_source_pages[0].get(
                'render_dpi', existing_manifest.get('default_render_dpi', 300)))
            previous_split = any(
                page.get('split_side') in ('left', 'right') or
                page.get('split_part') in ('a', 'b')
                for page in old_source_pages)
            rendering_changed = previous_dpi != render_dpi or previous_split != split_pages
            masks_path = project_manager.get_project_path(project_id, 'masks')
            cards_path = project_manager.get_project_path(project_id, 'cards')
            has_downstream_data = any(
                path and path.exists() and any(path.iterdir())
                for path in (masks_path, cards_path)
            )
            content_changed = True
            if pdf_filepath.exists():
                existing_hash = hashlib.sha256()
                with open(pdf_filepath, 'rb') as existing_file:
                    while True:
                        chunk = existing_file.read(1024 * 1024)
                        if not chunk:
                            break
                        existing_hash.update(chunk)
                uploaded_hash = hashlib.sha256()
                while True:
                    chunk = file.stream.read(1024 * 1024)
                    if not chunk:
                        break
                    uploaded_hash.update(chunk)
                file.stream.seek(0)
                content_changed = uploaded_hash.digest() != existing_hash.digest()
            if has_downstream_data and (rendering_changed or content_changed):
                return jsonify({
                    'error': (
                        'This PDF already has masks or cards created from its previous rendering. '
                        'Replacing its content or changing DPI/split-page mode would make their coordinates incorrect. '
                        'Use a new project, or clear and regenerate the downstream masks/cards first.'
                    ),
                    'success': False,
                }), 409
        # Get project images folder for output
        images_output_path = project_manager.get_project_path(project_id, 'images')
        
        # Process PDF and save images to project (use project name for image naming)
        print(f"Processing PDF to: {images_output_path}")
        project_name = secure_filename(project_metadata.get('project_name', 'project')) or project_id
        old_name = old_source_pages[0].get('image_name', '') if old_source_pages else ''
        old_base_match = re.match(r'(.+)_page_\d+[ab]?\.[^.]+$', old_name)
        render_base = old_base_match.group(1) if old_base_match else (
            project_name if not existing_sources else secure_filename(
                f"{project_name}_{Path(filename).stem}"))
        # Render into staging first. A corrupt PDF or interrupted render must
        # not overwrite a previously valid source PDF or leave half a page set.
        staging_root = Path(tempfile.mkdtemp(prefix='pypottery_pdf_',
                                             dir=app.config['UPLOAD_FOLDER']))
        staged_pdf = staging_root / filename
        staged_images = staging_root / 'images'
        staged_images.mkdir()
        try:
            file.save(staged_pdf)
            result = pdf_processor.process_pdf_to_folder(
                str(staged_pdf), str(staged_images), split_pages,
                project_name=render_base, render_dpi=render_dpi)
            print(f"Processing result: {result}")
            if str(result).startswith('Error processing PDF:'):
                raise MetadataLinkError(str(result))
            rendered_files = sorted(staged_images.glob(f'{render_base}_page_*.jpg'))
            if not rendered_files:
                raise MetadataLinkError('PDF rendering produced no page images')
            for rendered in rendered_files:
                os.replace(rendered, images_output_path / rendered.name)
            os.replace(staged_pdf, pdf_filepath)
        finally:
            shutil.rmtree(staging_root, ignore_errors=True)
        manifest = record_pdf_pages(
            project_manager.get_project_path(project_id), pdf_filepath,
            render_base, split_pages, profile_slug='hesban11', render_dpi=render_dpi
        )
        all_manifest_names = {page.get('image_name') for page in manifest.get('pages', [])}
        # A re-upload may switch split-page mode. Remove only superseded renders
        # after the replacement rendered successfully and its manifest was saved.
        for page in old_source_pages:
            old_image = images_output_path / Path(str(page.get('image_name', ''))).name
            if page.get('image_name') not in all_manifest_names and old_image.is_file():
                old_image.unlink()
        
        # Update project metadata
        image_count = project_manager.count_files(project_id, 'images')
        project_manager.update_workflow_status(project_id, {
            'pdf_processed': True,
            'pdf_count': len(list(pdf_source_path.glob('*.pdf'))),
            'images_extracted': image_count,
            'total_images': image_count
        })
        project_manager.update_settings(project_id, {'render_dpi': render_dpi})
        
        return jsonify({
            'message': f'PDF processed successfully. {image_count} images extracted.',
            'images_count': image_count,
            'render_dpi': render_dpi,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in PDF upload: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== MODEL APPLICATION ====================

@app.route('/api/model/apply', methods=['POST'])
def apply_model():
    """Apply model to images in a project"""
    try:
        data = request.json
        project_id = data.get('project_id')
        model = data.get('model')
        confidence = float(data.get('confidence', 0.5))
        diagnostic = data.get('diagnostic', False)
        kernel_size = int(data.get('kernel_size', 2))
        iterations = int(data.get('iterations', 10))
        excluded_images = data.get('excluded_images', [])
        
        if not project_id or not model:
            return jsonify({'error': 'Project and model are required', 'success': False}), 400
        
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Get project paths
        images_path = project_manager.get_project_path(project_id, 'images')
        masks_path = project_manager.get_project_path(project_id, 'masks')
        
        if not images_path or not images_path.exists():
            return jsonify({'error': 'Project images folder not found', 'success': False}), 404
        
        print(f"Applying model to project {project_id} with excluded_images: {excluded_images}")
        
        # Reset progress
        global model_progress
        model_progress = {
            'total': 0,
            'current': 0,
            'message': 'Starting...',
            'active': True
        }
        
        # Progress callback
        def update_progress(current, total, message):
            global model_progress
            model_progress['current'] = current
            model_progress['total'] = total
            model_progress['message'] = message
        
        # Run model in background thread
        def run_model():
            global model_progress
            try:
                result = model_processor.apply_model_to_project(
                    str(images_path),
                    str(masks_path),
                    model,
                    confidence,
                    diagnostic,
                    kernel_size,
                    iterations,
                    excluded_images,
                    progress_callback=update_progress
                )
                
                # Update project workflow status
                mask_count = project_manager.count_files(project_id, 'masks')
                project_manager.update_workflow_status(project_id, {
                    'model_applied': True,
                    'masks_extracted': mask_count
                })
                
                model_progress['message'] = result
                model_progress['active'] = False
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                model_progress['message'] = f'Error: {str(e)}'
                model_progress['active'] = False
        
        thread = threading.Thread(target=run_model)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'message': 'Model processing started',
            'success': True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/model/progress')
def get_model_progress():
    """Get current model processing progress"""
    global model_progress
    return jsonify(model_progress)


@app.route('/api/images/<folder>')
def get_folder_images(folder):
    """Get images in folder"""
    try:
        folder_path = PDFIMG_OUTPUT_DIR / folder
        if not folder_path.exists():
            return jsonify({'error': 'Folder not found', 'success': False}), 404
        
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Return image paths
        image_urls = [f'/api/image/{folder}/{img}' for img in images[:20]]  # Limit to 20 for preview
        
        return jsonify({
            'images': image_urls,
            'count': len(images),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/image/<folder>/<filename>')
def get_image(folder, filename):
    """Serve image file"""
    try:
        folder_path = PDFIMG_OUTPUT_DIR / folder
        return send_from_directory(folder_path, filename)
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 404


# ==================== ANNOTATION ====================

@app.route('/api/annotation/images/<folder>')
def get_annotation_images(folder):
    """Get list of images for annotation"""
    try:
        folder_path = PDFIMG_OUTPUT_DIR / folder
        if not folder_path.exists():
            return jsonify({'error': 'Folder not found', 'success': False}), 404
        
        images = sorted([f for f in os.listdir(folder_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        return jsonify({
            'images': images,
            'total': len(images),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/annotation/load', methods=['POST'])
def load_annotation():
    """Load image and existing annotation"""
    try:
        data = request.json
        folder = data.get('folder')
        image_name = data.get('image')
        
        if not folder or not image_name:
            return jsonify({'error': 'Folder and image are required', 'success': False}), 400
        
        image_path = PDFIMG_OUTPUT_DIR / folder / image_name
        if not image_path.exists():
            return jsonify({'error': 'Image not found', 'success': False}), 404
        
        # Load annotation data
        editor_data = annotation_processor.file_selection(str(image_path))
        
        # Convert to base64 for sending to client
        from PIL import Image
        import io
        
        img = Image.fromarray(editor_data['background'])
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Load mask if exists
        mask_base64 = None
        if editor_data['layers']:
            mask_img = Image.fromarray(editor_data['layers'][0])
            mask_buffer = io.BytesIO()
            mask_img.save(mask_buffer, format='PNG')
            mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode()
        
        return jsonify({
            'image': f'data:image/png;base64,{img_base64}',
            'mask': f'data:image/png;base64,{mask_base64}' if mask_base64 else None,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/annotation/save', methods=['POST'])
def save_annotation():
    """Save annotation mask"""
    try:
        data = request.json
        folder = data.get('folder')
        image_name = data.get('image')
        mask_data = data.get('mask')  # Base64 encoded mask
        
        if not all([folder, image_name, mask_data]):
            return jsonify({'error': 'Missing required data', 'success': False}), 400
        
        # Decode mask from base64
        from PIL import Image
        import io
        import numpy as np
        
        mask_bytes = base64.b64decode(mask_data.split(',')[1])
        mask_img = Image.open(io.BytesIO(mask_bytes))
        mask_array = np.array(mask_img)
        
        # Prepare editor_data format
        editor_data = {
            'layers': [mask_array]
        }
        
        # Save
        success = annotation_processor.save_annotation(folder, editor_data, image_name)
        
        return jsonify({
            'message': 'Mask saved successfully' if success else 'Failed to save mask',
            'success': success
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/annotation/extract', methods=['POST'])
def extract_masks():
    """Extract masks from annotations"""
    try:
        data = request.json
        folder = data.get('folder')
        
        if not folder:
            return jsonify({'error': 'Folder is required', 'success': False}), 400
        
        result = mask_extractor.extract_masks(folder)
        
        return jsonify({
            'message': result,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== TABULAR ====================

@app.route('/api/tabular/load', methods=['POST'])
def load_tabular_data():
    """Load tabular data for an image"""
    try:
        data = request.json
        folder = data.get('folder')
        img_num = int(data.get('img_num', 0))
        
        if not folder:
            return jsonify({'error': 'Folder is required', 'success': False}), 400
        
        # Get image and table data
        image_data, current_num, table_df, max_imgs = tabular_processor.image_selection(folder, img_num)
        
        # Convert image to base64 if exists
        img_base64 = None
        annotations = []
        if image_data and hasattr(image_data, 'value'):
            # Handle AnnotatedImage value
            img_array, annot_list = image_data.value
            from PIL import Image
            import io
            
            img = Image.fromarray(img_array)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            annotations = annot_list
        
        # Convert DataFrame to dict
        table_data = table_df.to_dict('records') if not table_df.empty else []
        
        return jsonify({
            'image': f'data:image/png;base64,{img_base64}' if img_base64 else None,
            'annotations': annotations,
            'table': table_data,
            'columns': list(table_df.columns) if not table_df.empty else [],
            'current': current_num,
            'total': max_imgs,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/tabular/save', methods=['POST'])
def save_tabular_data():
    """Save tabular data"""
    try:
        data = request.json
        folder = data.get('folder')
        table_data = data.get('table')
        
        if not folder or not table_data:
            return jsonify({'error': 'Missing required data', 'success': False}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(table_data)
        
        # Save
        tabular_processor.save_table(df, folder)
        
        return jsonify({
            'message': 'Table saved successfully',
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/tabular/add-column', methods=['POST'])
def add_column():
    """Add a new column to the table"""
    try:
        data = request.json
        column_name = data.get('column_name')
        table_data = data.get('table')
        
        if not column_name:
            return jsonify({'error': 'Column name is required', 'success': False}), 400
        
        df = pd.DataFrame(table_data)
        if column_name not in df.columns:
            df[column_name] = ""
        
        return jsonify({
            'table': df.to_dict('records'),
            'columns': list(df.columns),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/cards')
def get_project_cards(project_id):
    """Get list of card images for a project"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Get cards path
        cards_path = project_manager.get_project_path(project_id, 'cards')
        
        if not cards_path or not cards_path.exists():
            return jsonify({
                'cards': [],
                'total': 0,
                'success': True
            })
        
        # Get all card images
        def _natural_key(s):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]

        card_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        cards = sorted([f.name for f in cards_path.iterdir() 
                       if f.is_file() and f.suffix.lower() in card_extensions],
                      key=_natural_key)
        
        # Load classifications if available (check both cards and cards_modified)
        classifications = {}
        
        # Try cards_modified first (where classifications.csv is usually saved after processing)
        project_base_path = project_manager.get_project_path(project_id, 'cards')
        if project_base_path:
            project_root = project_base_path.parent
            cards_modified_path = project_root / 'cards_modified'
            classifications_csv = cards_modified_path / 'classifications.csv'
            
            if not classifications_csv.exists():
                # Fallback to cards folder
                classifications_csv = project_base_path / 'classifications.csv'
        else:
            classifications_csv = None
        
        if classifications_csv and classifications_csv.exists():
            try:
                import pandas as pd
                df = pd.read_csv(classifications_csv)
                print(f"Loaded classifications from {classifications_csv}, columns: {df.columns.tolist()}")
                
                # Create mapping from filename to type with normalization
                for _, row in df.iterrows():
                    # Try different column names for filename
                    filename = row.get('filename') or row.get('Filename') or row.get('mask_file') or row.get('id')
                    type_val = row.get('type') or row.get('Type')
                    
                    if filename and type_val:
                        # Normalize filename by removing path and keeping just the name
                        from pathlib import Path
                        filename_clean = Path(filename).name
                        classifications[filename_clean] = type_val
                        print(f"Mapped {filename_clean} -> {type_val}")
            except Exception as e:
                print(f"Error loading classifications: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("No classifications.csv found in cards or cards_modified")
        
        # Modified cards + exclusion list (both live in cards_modified/)
        cards_modified_dir = cards_path.parent / 'cards_modified'
        excluded_set = set(_read_excluded_cards(cards_modified_dir))

        # Create URLs and metadata for cards
        from PIL import Image as _PILImage
        card_data = []
        for card in cards:
            card_type = classifications.get(card, 'ENT')  # Default to ENT if not classified
            has_modified = (cards_modified_dir / card).exists()
            # Read pixel dimensions (lazy header read) for real-size grid sizing
            try:
                src = (cards_modified_dir / card) if has_modified else (cards_path / card)
                with _PILImage.open(src) as _im:
                    w, h = _im.size
            except Exception:
                w, h = 0, 0
            card_data.append({
                'url': f'/api/projects/{project_id}/card/{card}',
                'filename': card,
                'type': card_type,
                'width': w,
                'height': h,
                'has_modified': has_modified,
                'excluded': card in excluded_set,
            })

        return jsonify({
            'cards': card_data,
            'total': len(card_data),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/card/<filename>')
def serve_project_card(project_id, filename):
    """Serve a specific card image from project"""
    try:
        cards_path = project_manager.get_project_path(project_id, 'cards')
        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'Cards folder not found', 'success': False}), 404
        
        return send_from_directory(cards_path, filename)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 404


@app.route('/api/projects/<project_id>/card-modified/<filename>')
def serve_project_card_modified(project_id, filename):
    """Serve a specific modified card image from project"""
    try:
        cards_modified_path = project_manager.get_project_path(project_id, 'cards_modified')
        if not cards_modified_path or not cards_modified_path.exists():
            return jsonify({'error': 'Modified cards folder not found', 'success': False}), 404
        
        return send_from_directory(cards_modified_path, filename)
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 404


@app.route('/api/projects/<project_id>/tabular/load', methods=['POST'])
def load_project_tabular_data(project_id):
    """Load tabular data for a project - shows original image with bounding boxes"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        data = request.json
        img_num = int(data.get('img_num', 0))
        
        # Get project paths
        cards_path = project_manager.get_project_path(project_id, 'cards')
        images_path = project_manager.get_project_path(project_id, 'images')
        
        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'No cards found in project', 'success': False}), 404
        
        if not images_path or not images_path.exists():
            return jsonify({'error': 'No images found in project', 'success': False}), 404
        
        # Load annotation CSVs
        mask_info_path = cards_path / 'mask_info.csv'
        mask_info_annots_path = cards_path / 'mask_info_annots.csv'
        
        if not mask_info_path.exists() or not mask_info_annots_path.exists():
            return jsonify({'error': 'Annotation CSV files not found', 'success': False}), 404
        
        # Read CSV files
        df_info = pd.read_csv(mask_info_path, dtype=str, keep_default_na=False)
        df_info = migrate_linkage_columns(df_info, drop_legacy=False)
        df_annots = pd.read_csv(mask_info_annots_path)
        linkage_state = load_linkage_state(project_manager.get_project_path(project_id))
        staged_numbers = {
            Path(str(drawing.get('mask_file', ''))).stem: str(drawing.get('vessel_number', '')).strip()
            for figure in linkage_state.get('figures', [])
            for drawing in figure.get('drawings', [])
            if drawing.get('mask_file')
        }
        
        # Add image_name and ID columns to annotations
        df_annots['image_name'] = df_annots['mask_file'].apply(
            lambda x: x.split('_mask_layer_')[0] if isinstance(x, str) and '_mask_layer_' in x else '')
        df_annots['ID'] = df_annots['mask_file'].apply(
            lambda x: x.split('layer_')[1].replace('.png', '') if isinstance(x, str) and 'layer_' in x else '0')
        
        # Get list of unique images
        unique_images = sorted(df_annots['image_name'].unique())
        
        if not unique_images:
            return jsonify({'error': 'No images found in annotations', 'success': False}), 404
        
        # Validate image number
        img_num = max(0, min(img_num, len(unique_images) - 1))
        current_image_name = unique_images[img_num]
        
        # Find original image file
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        original_image_path = None
        
        for ext in image_extensions:
            candidate = images_path / f"{current_image_name}{ext}"
            if candidate.exists():
                original_image_path = candidate
                break
        
        if not original_image_path:
            return jsonify({'error': f'Original image not found: {current_image_name}', 'success': False}), 404
        
        # Load and prepare image with annotations
        from PIL import Image
        import io

        with Image.open(original_image_path) as img:
            # Keep original size for bbox math
            original_size = img.size  # (width, height)

            # Create a reasonably sized preview but preserve aspect ratio (no rotation)
            img.thumbnail((1200, 1200))

            # Compute scale factors
            scale_x = img.size[0] / original_size[0]
            scale_y = img.size[1] / original_size[1]

            image_array = np.asarray(img, dtype=np.uint8)
        
        # Get annotations for this image
        image_annots = df_annots[df_annots['image_name'] == current_image_name]
        
        # Create scaled annotations list
        annotations = []
        for _, row in image_annots.iterrows():
            try:
                # Parse bbox string "(x1, y1, x2, y2)"
                bbox_str = str(row.get('bbox', '')).strip('()')
                coords = [int(x.strip()) for x in bbox_str.split(',')]

                # Scale coordinates (no rotation - keep original orientation)
                scaled_bbox = [
                    int(coords[0] * scale_x),
                    int(coords[1] * scale_y),
                    int(coords[2] * scale_x),
                    int(coords[3] * scale_y)
                ]

                mask_key = str(row.get('mask_file', '')).removesuffix('.png')
                info_match = df_info[df_info['mask_file'].astype(str).map(
                    lambda value: Path(value).stem) == mask_key]
                vessel_number = '' if info_match.empty else str(info_match.iloc[0].get('No.', '')).strip()
                vessel_number = vessel_number or staged_numbers.get(mask_key, '')
                annotations.append({
                    'bbox': scaled_bbox,
                    'label': vessel_number or '?',
                    'row_key': mask_key,
                })

            except Exception as e:
                print(f"Error processing annotation: {e}")
                continue
        
        # Convert image to base64
        buffer = io.BytesIO()
        Image.fromarray(image_array).save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
    # Prepare table data from mask_info
        df_subset = df_info[df_info['file'] == current_image_name].copy()
        
        if not df_subset.empty:
            if 'No.' not in df_subset.columns:
                df_subset['No.'] = ''
            df_subset['No.'] = [
                str(number).strip() or staged_numbers.get(Path(str(mask_file)).stem, '')
                for number, mask_file in zip(df_subset['No.'], df_subset['mask_file'])
            ]
            df_subset['No.'] = df_subset['No.'].replace('', '?')
            # Keep mask_file in row JSON as a stable private key, but do not
            # expose it as a table column.
            drop_cols = [col for col in ['file'] if col in df_subset.columns]
            if drop_cols:
                df_subset = df_subset.drop(columns=drop_cols)
            
            table_data = df_subset.to_dict('records')
            columns = ['No.'] + [col for col in df_subset.columns if col not in {'No.', 'mask_file'}]
        else:
            # Create empty table structure
            table_data = []
            columns = ['No.', 'Notes']
        
        # Build image list with reviewed flags
        project_meta = project_metadata
        reviewed_list = project_meta.get('workflow_status', {}).get('reviewed_images', []) if project_meta else []
        image_list = [{'image_name': name, 'reviewed': (name in reviewed_list)} for name in unique_images]

        # Prepare full-resolution image URL for zoom (let frontend fetch it when user requests zoom)
        full_image_url = None
        for ext in image_extensions:
            candidate = images_path / f"{current_image_name}{ext}"
            if candidate.exists():
                full_image_url = f'/api/projects/{project_id}/image/{candidate.name}'
                break

        return jsonify({
            'image': f'data:image/png;base64,{img_base64}',
            'annotations': annotations,
            'table': table_data,
            'columns': columns,
            'current': img_num,
            'total': len(unique_images),
            'image_name': current_image_name,
            'image_list': image_list,
            'is_reviewed': (current_image_name in reviewed_list),
            'full_image_url': full_image_url,
            'success': True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/tabular/save', methods=['POST'])
def save_project_tabular_data(project_id):
    """Save tabular data for a project - updates mask_info.csv"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        data = request.json
        table_data = data.get('table')
        image_name = data.get('image_name')  # Current image being edited
        
        if not table_data:
            return jsonify({'error': 'Missing table data', 'success': False}), 400
        
        # Get cards folder path
        cards_path = project_manager.get_project_path(project_id, 'cards')
        
        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'Cards folder not found', 'success': False}), 404
        
        # Convert to DataFrame
        df_new = pd.DataFrame(table_data)
        if 'No.' in df_new.columns:
            df_new['No.'] = df_new['No.'].replace('?', '')
        
        # Load existing mask_info.csv
        csv_path = cards_path / 'mask_info.csv'
        
        if csv_path.exists():
            try:
                df_existing = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
                df_existing = migrate_linkage_columns(df_existing, drop_legacy=True)
                
                # Remove old data for current image
                if image_name and 'file' in df_existing.columns:
                    df_existing = df_existing[df_existing['file'] != image_name]
                
                # Add file and mask_file columns to new data if not present
                if 'file' not in df_new.columns and image_name:
                    df_new['file'] = image_name
                
                if 'mask_file' not in df_new.columns:
                    return jsonify({'error': 'Stable card identifiers are missing; reload this page',
                                    'success': False}), 409
                
                # Combine old and new data
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                
                # Save combined data
                df_combined.to_csv(csv_path, index=False)
                
            except Exception as e:
                print(f"Warning: Could not merge with existing CSV: {e}")
                # Just save new data
                df_new.to_csv(csv_path, index=False)
        else:
            # Add required columns if missing
            if 'file' not in df_new.columns and image_name:
                df_new['file'] = image_name
            
            if 'mask_file' not in df_new.columns:
                return jsonify({'error': 'Stable card identifiers are missing; reload this page',
                                'success': False}), 409
            
            df_new.to_csv(csv_path, index=False)
        
        # If vessel numbers are edited in the general table, keep the staged
        # linkage state and its validation in sync immediately.
        if {'mask_file', 'No.'}.issubset(df_new.columns):
            state = load_linkage_state(project_manager.get_project_path(project_id))
            submitted = {Path(str(row['mask_file'])).stem: str(row['No.']).strip()
                         for _, row in df_new.iterrows()}
            state_changed = False
            profile = get_profile(state.get('profile', 'hesban11'))
            for figure in state.get('figures', []):
                figure_changed = False
                for drawing in figure.get('drawings', []):
                    key = Path(str(drawing.get('mask_file', ''))).stem
                    if key in submitted and drawing.get('vessel_number', '') != submitted[key]:
                        drawing['vessel_number'] = submitted[key]
                        figure_changed = state_changed = True
                if figure_changed:
                    figure['review_status'] = 'pending'
                    validate_figure(figure, profile)
            if state_changed:
                save_linkage_state(project_manager.get_project_path(project_id), state)
                project_manager.update_workflow_status(project_id, linkage_totals(state))

        print(f"Saved tabular data to: {csv_path}")
        
        return jsonify({
            'message': 'Table saved successfully',
            'success': True
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# --------------------------------------------------------------------------
# AI extraction helpers (shared by single + batch bibliographic extraction)
# --------------------------------------------------------------------------

def _normalize_field_key(s):
    """Loose key for matching column names regardless of case/spacing/punctuation."""
    return _re.sub(r'[^a-z0-9]', '', str(s).lower())


def _canonical_fields_from_prompt(prompt_suffix):
    """Field names the user wrote inside [square brackets] become EXACT column names.

    e.g. "Add also [Scale], which is on the bottom-left" -> ["Scale"]. This keeps a
    single, stable column across every page/iteration instead of the model inventing
    variants (scale, Scale, scala...).
    """
    if not prompt_suffix:
        return []
    out = []
    for f in _re.findall(r'\[([^\[\]]+)\]', prompt_suffix):
        f = f.strip()
        if f and f not in out:
            out.append(f)
    return out


def _extra_fields_instruction(canonical_fields):
    """Prompt fragment telling the model which extra keys to use, verbatim."""
    if canonical_fields:
        keys = ", ".join(f'"{c}"' for c in canonical_fields)
        return (
            "In addition to the four standard fields, also extract these fields and use "
            "these EXACT JSON keys, spelled verbatim — do NOT translate, rename, pluralise "
            f"or change the case: {keys}.\n"
            "Use the same key on every drawing; if a value is not present, use null.\n"
        )
    return (
        "If the context above asks you to extract additional fields beyond the four "
        "standard ones, include them in each drawing's object using a concise snake_case "
        "key (e.g. 'material', 'ceramic_class').\n"
    )


def _canonicalize_keys(values, canonical_fields):
    """Rename model-returned keys to the exact canonical spelling when they match."""
    if not canonical_fields or not isinstance(values, dict):
        return values
    lookup = {_normalize_field_key(c): c for c in canonical_fields}
    out = {}
    for k, v in values.items():
        out[lookup.get(_normalize_field_key(k), k)] = v
    return out


def _parse_ai_json(raw_response):
    """Extract and parse the JSON object from an LLM response, tolerantly.

    Handles markdown fences, smart quotes, trailing commas, and — importantly —
    responses truncated by the token limit, by trimming back to the last complete
    drawing object so partial results are still usable. Raises ValueError if no
    usable JSON can be recovered.
    """
    if not raw_response or not raw_response.strip():
        raise ValueError("empty response")

    text = raw_response.strip()
    if text.startswith("```"):
        text = _re.sub(r'^```[a-zA-Z]*\n?', '', text)
        text = _re.sub(r'\n?```\s*$', '', text).strip()
    text = text.replace('“', '"').replace('”', '"').replace('’', "'")

    start = text.find('{')
    if start == -1:
        raise ValueError("no JSON object found in response")

    # Walk to the matching closing brace, respecting string literals/escapes
    depth = 0
    in_str = False
    esc = False
    end = None
    last_container_close = None
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                last_container_close = i
                if depth == 0:
                    end = i + 1
                    break
            elif ch == ']':
                last_container_close = i

    candidate = text[start:end] if end else text[start:]

    # Attempt 1: as-is
    try:
        return _json.loads(candidate)
    except _json.JSONDecodeError:
        pass

    # Attempt 2: strip trailing commas before } or ]
    repaired = _re.sub(r',\s*([}\]])', r'\1', candidate)
    try:
        return _json.loads(repaired)
    except _json.JSONDecodeError:
        pass

    # Attempt 3: response was truncated — keep complete drawing objects only
    if last_container_close is not None:
        trimmed = text[start:last_container_close + 1].rstrip().rstrip(',')
        stack = []
        in_str = False
        esc = False
        for ch in trimmed:
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch in '{[':
                stack.append(ch)
            elif ch == '}' and stack and stack[-1] == '{':
                stack.pop()
            elif ch == ']' and stack and stack[-1] == '[':
                stack.pop()
        trimmed += ''.join('}' if opener == '{' else ']' for opener in reversed(stack))
        trimmed = _re.sub(r',\s*([}\]])', r'\1', trimmed)
        return _json.loads(trimmed)  # raises ValueError-compatible on failure

    raise ValueError("could not parse JSON from response")


@app.route('/api/projects/<project_id>/tabular/ai-bibliographic', methods=['POST'])
def ai_extract_bibliographic(project_id):
    """Use Gemma 4 E2B-it to extract bibliographic info (tavola, figura, numero)
    from the original page image using bounding box coordinates."""
    try:
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404

        data = request.json
        img_num = int(data.get('img_num', 0))
        prompt_suffix = str(data.get('prompt_suffix', '')).strip()
        ai_backend = str(data.get('ai_backend', 'local')).strip()
        openrouter_api_key = str(data.get('openrouter_api_key', '')).strip()
        openrouter_model = str(data.get('openrouter_model', 'deepseek/deepseek-v4-flash')).strip()
        numbers_from_crops = bool(data.get('numbers_from_crops', False))

        cards_path = project_manager.get_project_path(project_id, 'cards')
        images_path = project_manager.get_project_path(project_id, 'images')

        mask_info_path = cards_path / 'mask_info.csv'
        mask_info_annots_path = cards_path / 'mask_info_annots.csv'

        if not mask_info_path.exists() or not mask_info_annots_path.exists():
            return jsonify({'error': 'Annotation CSV files not found', 'success': False}), 404

        df_info = pd.read_csv(mask_info_path).fillna('')
        df_annots = pd.read_csv(mask_info_annots_path)

        df_annots['image_name'] = df_annots['mask_file'].apply(
            lambda x: x.split('_mask_layer_')[0] if isinstance(x, str) and '_mask_layer_' in x else '')
        df_annots['ID'] = df_annots['mask_file'].apply(
            lambda x: x.split('layer_')[1].replace('.png', '') if isinstance(x, str) and 'layer_' in x else '0')

        unique_images = sorted(df_annots['image_name'].unique())
        if not unique_images:
            return jsonify({'error': 'No images found in annotations', 'success': False}), 404

        img_num = max(0, min(img_num, len(unique_images) - 1))
        current_image_name = unique_images[img_num]

        # Find original image file
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        original_image_path = None
        for ext in image_extensions:
            candidate = images_path / f"{current_image_name}{ext}"
            if candidate.exists():
                original_image_path = candidate
                break

        if not original_image_path:
            return jsonify({'error': f'Original image not found: {current_image_name}', 'success': False}), 404

        # Load image at high resolution for OCR (max 2400px on longest side)
        from PIL import Image as PILImage
        with PILImage.open(original_image_path) as img:
            orig_w, orig_h = img.size
            img_copy = img.copy()
            img_copy.thumbnail((2400, 2400), PILImage.LANCZOS)
            scale_x = img_copy.size[0] / orig_w
            scale_y = img_copy.size[1] / orig_h
            page_image = img_copy.convert('RGB')

        # Build bbox list and visually annotate the image
        image_annots = df_annots[df_annots['image_name'] == current_image_name]
        bbox_lines = []
        bbox_data  = []   # (label, x1, y1, x2, y2) for visual annotation
        for _, row in image_annots.iterrows():
            try:
                bbox_str = str(row.get('bbox', '')).strip('()')
                coords = [int(x.strip()) for x in bbox_str.split(',')]
                sx1 = int(coords[0] * scale_x)
                sy1 = int(coords[1] * scale_y)
                sx2 = int(coords[2] * scale_x)
                sy2 = int(coords[3] * scale_y)
                bbox_lines.append(f"- ID {row['ID']}: [{sx1}, {sy1}, {sx2}, {sy2}]")
                bbox_data.append((str(row['ID']), sx1, sy1, sx2, sy2))
            except Exception:
                continue

        if not bbox_lines:
            return jsonify({'error': 'No valid annotations found for this page', 'success': False}), 404

        # Use letter labels (A, B, C...) on drawn boxes so the model cannot
        # confuse the annotation label with the publication's own catalogue number.
        import string as _string
        _LETTERS = _string.ascii_uppercase
        letter_map = {}          # letter -> actual row ID string
        letter_bbox_data = []
        for _i, (_aid, _x1, _y1, _x2, _y2) in enumerate(bbox_data):
            _lbl = _LETTERS[_i] if _i < len(_LETTERS) else f"Z{_i}"
            letter_map[_lbl] = _aid
            letter_bbox_data.append((_lbl, _x1, _y1, _x2, _y2))

        page_image = _annotate_image_with_bboxes(page_image, letter_bbox_data)

        letter_list_str = ', '.join(letter_map.keys())
        prompt = (
            "This is a page from an archaeological publication about pottery.\n"
            f"There are {len(letter_bbox_data)} pottery drawings on this page. "
            "Each drawing is VISUALLY MARKED with an ORANGE BOUNDING BOX. "
            "The orange letter (A, B, C...) at the top of each box is a software "
            "annotation only — it is NOT a number from the publication.\n\n"
            f"Your JSON response MUST use EXACTLY these letter keys: {letter_list_str}\n"
            "Do NOT use numbers as keys. Each key must be one of the letters listed above.\n\n"
            "For EACH drawing, extract:\n"
            '- "page": the page number printed on the publication page. '
            'Look at the edges and corners of the full image. Same for all drawings.\n'
            '- "plate": plate/table identifier (e.g. "Tav. III", "Pl. 12"). '
            'Usually at the top or bottom edge. Shared by all drawings.\n'
            '- "figure": figure identifier for the plate (e.g. "Fig. 3", "Abb. 5"). '
            'Usually at top or bottom of the image.\n'
            '- "number": the small catalogue number PRINTED IN THE ORIGINAL PUBLICATION '
            'near or inside the drawing inside the orange box. '
            'It is a digit or short alphanumeric (e.g. "1", "3", "14", "2a", "7b") '
            'that appears as part of the publication layout, NOT the orange letter label.\n\n'
        )
        canonical_fields = _canonical_fields_from_prompt(prompt_suffix)
        if prompt_suffix:
            prompt += f"Additional context from the user:\n{prompt_suffix}\n\n"
        prompt += (
            f"Respond ONLY with a valid JSON object using EXACTLY these letter keys: {letter_list_str}\n"
            + _extra_fields_instruction(canonical_fields) +
            "Example for two drawings labelled A and B:\n"
            '{"A": {"page": "45", "plate": "Tav. III", "figure": "Fig. 5", "number": "1"}, '
            '"B": {"page": "45", "plate": "Tav. III", "figure": "Fig. 5", "number": "2a"}}\n'
            "If a value is not found, use null."
        )

        # Token budget scales with the number of drawings so large pages (30+
        # drawings) are not truncated mid-JSON.
        max_tokens = min(4096, max(512, len(letter_bbox_data) * 70 + 256))

        # ---- Run inference (local Gemma or OpenRouter) ----
        if ai_backend == 'openrouter':
            if not openrouter_api_key:
                return jsonify({'error': 'OpenRouter API key is required', 'success': False}), 400
            raw_response = call_openrouter_ai(page_image, prompt, openrouter_api_key, openrouter_model, max_tokens)
            print(f"[AI OpenRouter] Raw response: {raw_response[:300]}")
        else:
            # Load Gemma 4 E2B-it and run inference
            model, processor = load_gemma_model()

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": page_image},
                    {"type": "text", "text": prompt}
                ]
            }]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = processor(text=text, images=[page_image], return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=64,
                    do_sample=True
                )

            raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
            print(f"[AI] Raw response: {raw_response[:300]}")

        # Robustly parse JSON (handles fences, smart quotes, truncation)
        try:
            ai_result = _parse_ai_json(raw_response)
        except Exception as parse_err:
            return jsonify({
                'error': f'Model did not return valid JSON ({parse_err}). Response: {raw_response[:500]}',
                'success': False
            }), 500

        # Force the user's bracketed field names to be exact columns and
        # normalise model key variants onto them.
        if canonical_fields:
            ai_result = {k: _canonicalize_keys(v, canonical_fields)
                         for k, v in ai_result.items()}
            for col in canonical_fields:
                if col not in df_info.columns:
                    df_info[col] = ''

        # Ensure all columns returned by the model exist in df_info
        # (handles both the 4 standard fields and any extra user-requested ones)
        all_field_keys = set()
        for _vals in ai_result.values():
            if isinstance(_vals, dict):
                all_field_keys.update(_vals.keys())
        for col in all_field_keys:
            if col not in df_info.columns:
                df_info[col] = ''

        # Write extracted values into df_info rows for current image.
        # Model response uses letter keys (A, B, C...); remap to actual row IDs.
        for letter, values in ai_result.items():
            if not isinstance(values, dict):
                continue
            mask_id = letter_map.get(letter)
            if mask_id is None:
                print(f"[AI] Warning: unexpected key {letter!r} in response, skipping")
                continue
            exact_mask_file = f"{current_image_name}_mask_layer_{mask_id}.png"
            row_mask = df_info['mask_file'] == exact_mask_file
            if not row_mask.any():
                exact_no_ext = f"{current_image_name}_mask_layer_{mask_id}"
                row_mask = df_info['mask_file'].apply(
                    lambda x: str(x).replace('.png', '') == exact_no_ext
                )
            if row_mask.any():
                for col, val in values.items():
                    if val is not None:
                        df_info.loc[row_mask, col] = str(val)
            else:
                print(f"[AI] Warning: no row found in df_info for mask_id={mask_id!r}, expected file={exact_mask_file!r}")

        # Optionally re-read inventory numbers from per-drawing crops (more
        # reliable for tiny numbers), overwriting the globally-read 'number'.
        if numbers_from_crops:
            try:
                _read_numbers_from_crops(
                    original_image_path, image_annots, current_image_name,
                    df_info, ai_backend, openrouter_api_key, openrouter_model)
            except Exception as _ne:
                print(f"[AI] numbers-from-crops failed: {_ne}")

        df_info.to_csv(mask_info_path, index=False)

        # Return updated table for current image
        df_subset = df_info[df_info['file'] == current_image_name].copy()
        if not df_subset.empty:
            df_subset['ID'] = df_subset['mask_file'].apply(
                lambda x: x.split('layer_')[1] if isinstance(x, str) and 'layer_' in x else '0')
            drop_cols = [col for col in ['mask_file', 'file'] if col in df_subset.columns]
            if drop_cols:
                df_subset = df_subset.drop(columns=drop_cols)
            columns_order = ['ID'] + [col for col in df_subset.columns if col != 'ID']
            df_subset = df_subset[columns_order]
            table_data = df_subset.to_dict('records')
            columns = list(df_subset.columns)
        else:
            table_data = []
            columns = ['ID', 'page', 'plate', 'figure', 'number']

        return jsonify({
            'success': True,
            'table': table_data,
            'columns': columns,
            'ai_result': ai_result
        })

    except VisionUnsupportedError as e:
        return jsonify({'error': str(e), 'success': False, 'vision_unsupported': True}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/tabular/ai-bibliographic-batch', methods=['POST'])
def ai_extract_bibliographic_batch(project_id):
    """Run Gemma 4 E2B-it AI extraction on ALL images in the project (batch mode).
    Progress is streamed via the global operation_progress dict so the frontend
    can poll /api/progress."""
    try:
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404

        data = request.json or {}
        prompt_suffix = str(data.get('prompt_suffix', '')).strip()
        ai_backend = str(data.get('ai_backend', 'local')).strip()
        openrouter_api_key = str(data.get('openrouter_api_key', '')).strip()
        openrouter_model = str(data.get('openrouter_model', 'deepseek/deepseek-v4-flash')).strip()
        numbers_from_crops = bool(data.get('numbers_from_crops', False))

        if ai_backend == 'openrouter' and not openrouter_api_key:
            return jsonify({'error': 'OpenRouter API key is required', 'success': False}), 400

        cards_path = project_manager.get_project_path(project_id, 'cards')
        images_path = project_manager.get_project_path(project_id, 'images')

        mask_info_path = cards_path / 'mask_info.csv'
        mask_info_annots_path = cards_path / 'mask_info_annots.csv'

        if not mask_info_path.exists() or not mask_info_annots_path.exists():
            return jsonify({'error': 'Annotation CSV files not found', 'success': False}), 404

        df_info = pd.read_csv(mask_info_path).fillna('')
        df_annots = pd.read_csv(mask_info_annots_path)

        df_annots['image_name'] = df_annots['mask_file'].apply(
            lambda x: x.split('_mask_layer_')[0] if isinstance(x, str) and '_mask_layer_' in x else '')
        df_annots['ID'] = df_annots['mask_file'].apply(
            lambda x: x.split('layer_')[1].replace('.png', '') if isinstance(x, str) and 'layer_' in x else '0')

        unique_images = sorted(df_annots['image_name'].unique())
        if not unique_images:
            return jsonify({'error': 'No images found in annotations', 'success': False}), 404

        # Ensure English columns exist
        for col in ['page', 'plate', 'figure', 'number']:
            if col not in df_info.columns:
                df_info[col] = ''

        total = len(unique_images)
        update_operation_progress('ai_batch', 0, total, 'Loading AI model...')

        # Load local model only if needed (skip for OpenRouter)
        model, processor = (None, None)
        if ai_backend != 'openrouter':
            model, processor = load_gemma_model()

        errors = []
        for idx, current_image_name in enumerate(unique_images):
            update_operation_progress('ai_batch', idx, total,
                                      f'Processing image {idx + 1}/{total}: {current_image_name}')

            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            original_image_path = None
            for ext in image_extensions:
                candidate = images_path / f"{current_image_name}{ext}"
                if candidate.exists():
                    original_image_path = candidate
                    break

            if not original_image_path:
                errors.append(f'Image not found: {current_image_name}')
                continue

            from PIL import Image as PILImage
            with PILImage.open(original_image_path) as img:
                orig_w, orig_h = img.size
                img_copy = img.copy()
                img_copy.thumbnail((2400, 2400), PILImage.LANCZOS)
                scale_x = img_copy.size[0] / orig_w
                scale_y = img_copy.size[1] / orig_h
                page_image = img_copy.convert('RGB')

            image_annots = df_annots[df_annots['image_name'] == current_image_name]
            bbox_lines = []
            bbox_data  = []   # (label, x1, y1, x2, y2) for visual annotation
            for _, row in image_annots.iterrows():
                try:
                    bbox_str = str(row.get('bbox', '')).strip('()')
                    coords = [int(x.strip()) for x in bbox_str.split(',')]
                    sx1 = int(coords[0] * scale_x)
                    sy1 = int(coords[1] * scale_y)
                    sx2 = int(coords[2] * scale_x)
                    sy2 = int(coords[3] * scale_y)
                    bbox_lines.append(f"- ID {row['ID']}: [{sx1}, {sy1}, {sx2}, {sy2}]")
                    bbox_data.append((str(row['ID']), sx1, sy1, sx2, sy2))
                except Exception:
                    continue

            if not bbox_lines:
                errors.append(f'No annotations for: {current_image_name}')
                continue

            # Use letter labels (A, B, C...) on drawn boxes so the model cannot
            # confuse the annotation label with the publication's own catalogue number.
            import string as _string
            _LETTERS = _string.ascii_uppercase
            letter_map = {}          # letter -> actual row ID string
            letter_bbox_data = []
            for _i, (_aid, _x1, _y1, _x2, _y2) in enumerate(bbox_data):
                _lbl = _LETTERS[_i] if _i < len(_LETTERS) else f"Z{_i}"
                letter_map[_lbl] = _aid
                letter_bbox_data.append((_lbl, _x1, _y1, _x2, _y2))

            page_image = _annotate_image_with_bboxes(page_image, letter_bbox_data)

            letter_list_str = ', '.join(letter_map.keys())
            prompt = (
                "This is a page from an archaeological publication about pottery.\n"
                f"There are {len(letter_bbox_data)} pottery drawings on this page. "
                "Each drawing is VISUALLY MARKED with an ORANGE BOUNDING BOX. "
                "The orange letter (A, B, C...) at the top of each box is a software "
                "annotation only — it is NOT a number from the publication.\n\n"
                f"Your JSON response MUST use EXACTLY these letter keys: {letter_list_str}\n"
                "Do NOT use numbers as keys. Each key must be one of the letters listed above.\n\n"
                "For EACH drawing, extract:\n"
                '- "page": the page number printed on the publication page. '
                'Look at the edges and corners of the full image. Same for all drawings.\n'
                '- "plate": plate/table identifier (e.g. "Tav. III", "Pl. 12"). '
                'Usually at the top or bottom edge. Shared by all drawings.\n'
                '- "figure": figure identifier for the plate (e.g. "Fig. 3", "Abb. 5"). '
                'Usually at top or bottom of the image.\n'
                '- "number": the small catalogue number PRINTED IN THE ORIGINAL PUBLICATION '
                'near or inside the drawing inside the orange box. '
                'It is a digit or short alphanumeric (e.g. "1", "3", "14", "2a", "7b") '
                'that appears as part of the publication layout, NOT the orange letter label.\n\n'
            )
            canonical_fields = _canonical_fields_from_prompt(prompt_suffix)
            if prompt_suffix:
                prompt += f"Additional context from the user:\n{prompt_suffix}\n\n"
            prompt += (
                f"Respond ONLY with a valid JSON object using EXACTLY these letter keys: {letter_list_str}\n"
                + _extra_fields_instruction(canonical_fields) +
                "Example for two drawings labelled A and B:\n"
                '{"A": {"page": "45", "plate": "Tav. III", "figure": "Fig. 5", "number": "1"}, '
                '"B": {"page": "45", "plate": "Tav. III", "figure": "Fig. 5", "number": "2a"}}\n'
                "If a value is not found, use null."
            )

            max_tokens = min(4096, max(512, len(letter_bbox_data) * 70 + 256))

            # ---- Run inference (local Gemma or OpenRouter) ----
            if ai_backend == 'openrouter':
                try:
                    raw_response = call_openrouter_ai(page_image, prompt, openrouter_api_key, openrouter_model, max_tokens)
                    print(f"[AI Batch OpenRouter] {current_image_name} response: {raw_response[:200]}")
                except Exception as _or_err:
                    errors.append(f'OpenRouter error for {current_image_name}: {_or_err}')
                    continue
            else:
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "image", "image": page_image},
                        {"type": "text", "text": prompt}
                    ]
                }]

                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                inputs = processor(text=text, images=[page_image], return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[-1]

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        temperature=1.0,
                        top_p=0.95,
                        top_k=64,
                        do_sample=True
                    )

                raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
                print(f"[AI Batch] {current_image_name} response: {raw_response[:200]}")

            try:
                ai_result = _parse_ai_json(raw_response)
            except Exception as _pe:
                errors.append(f'Invalid JSON for {current_image_name}: {_pe}')
                continue

            # Force the user's bracketed field names to be exact columns
            if canonical_fields:
                ai_result = {k: _canonicalize_keys(v, canonical_fields)
                             for k, v in ai_result.items()}
                for col in canonical_fields:
                    if col not in df_info.columns:
                        df_info[col] = ''

            # Ensure any extra columns the model returned exist in df_info
            for _vals in ai_result.values():
                if isinstance(_vals, dict):
                    for col in _vals.keys():
                        if col not in df_info.columns:
                            df_info[col] = ''

            # Remap letter keys (A, B, C...) back to actual row IDs
            for letter, values in ai_result.items():
                if not isinstance(values, dict):
                    continue
                mask_id = letter_map.get(letter)
                if mask_id is None:
                    print(f"[AI Batch] Warning: unexpected key {letter!r} in response, skipping")
                    continue
                exact_mask_file = f"{current_image_name}_mask_layer_{mask_id}.png"
                row_mask = df_info['mask_file'] == exact_mask_file
                if not row_mask.any():
                    exact_no_ext = f"{current_image_name}_mask_layer_{mask_id}"
                    row_mask = df_info['mask_file'].apply(
                        lambda x: str(x).replace('.png', '') == exact_no_ext
                    )
                if row_mask.any():
                    for col, val in values.items():
                        if val is not None:
                            df_info.loc[row_mask, col] = str(val)

            # Per-image: optionally re-read inventory numbers from crops,
            # overwriting the globally-read 'number' with a zoomed-in reading.
            if numbers_from_crops:
                try:
                    _read_numbers_from_crops(
                        original_image_path, image_annots, current_image_name,
                        df_info, ai_backend, openrouter_api_key, openrouter_model,
                        model=model, processor=processor)
                except Exception as _ne:
                    print(f"[AI Batch] numbers-from-crops failed for {current_image_name}: {_ne}")

        df_info.to_csv(mask_info_path, index=False)
        clear_operation_progress()

        return jsonify({
            'success': True,
            'processed': total,
            'errors': errors
        })

    except VisionUnsupportedError as e:
        clear_operation_progress()
        return jsonify({'error': str(e), 'success': False, 'vision_unsupported': True}), 400
    except Exception as e:
        import traceback
        traceback.print_exc()
        clear_operation_progress()
        return jsonify({'error': str(e), 'success': False}), 500


def _read_numbers_from_crops(original_image_path, image_annots, current_image_name,
                             df_info, ai_backend, openrouter_api_key, openrouter_model,
                             model=None, processor=None):
    """Overwrite the 'number' column by reading each drawing's inventory number
    from a zoomed-in crop of the FULL-resolution page (one focused AI call each).

    Inventory numbers are tiny and printed right next to the drawing, so they are
    easily missed on the full busy page. Mutates ``df_info`` in place and returns
    how many numbers were updated.
    """
    from PIL import Image as PILImage

    if ai_backend != 'openrouter' and model is None:
        model, processor = load_gemma_model()

    full_img = PILImage.open(original_image_path).convert('RGB')
    W, H = full_img.size
    if 'number' not in df_info.columns:
        df_info['number'] = ''

    number_prompt = (
        "This is a cropped detail from an archaeological pottery catalogue. "
        "It shows one pottery drawing and the area around it. Find the small "
        "catalogue/inventory NUMBER printed next to or under the drawing — it is "
        "a digit or short alphanumeric such as '1', '14', '2a', '7b'. "
        "Reply with ONLY that number and nothing else. If there is no number, reply 'null'."
    )

    updated = 0
    for _, row in image_annots.iterrows():
        try:
            bbox_str = str(row.get('bbox', '')).strip('()')
            x1, y1, x2, y2 = [int(v.strip()) for v in bbox_str.split(',')]
        except Exception:
            continue

        # Expand the box with margin ("gioco"); more below where the number sits.
        bw, bh = x2 - x1, y2 - y1
        mx = int(bw * 0.35) + 15
        my_top = int(bh * 0.25) + 15
        my_bot = int(bh * 0.50) + 15
        cx1, cy1 = max(0, x1 - mx), max(0, y1 - my_top)
        cx2, cy2 = min(W, x2 + mx), min(H, y2 + my_bot)
        if cx2 <= cx1 or cy2 <= cy1:
            continue

        crop = full_img.crop((cx1, cy1, cx2, cy2))
        longest = max(crop.size)
        if longest < 768:
            factor = 768.0 / longest
            crop = crop.resize((int(crop.size[0] * factor), int(crop.size[1] * factor)),
                               PILImage.LANCZOS)

        try:
            if ai_backend == 'openrouter':
                raw = call_openrouter_ai(crop, number_prompt, openrouter_api_key,
                                         openrouter_model, max_tokens=20)
            else:
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": crop},
                    {"type": "text", "text": number_prompt}]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                inputs = processor(text=text, images=[crop], return_tensors="pt").to(model.device)
                input_len = inputs["input_ids"].shape[-1]
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
                raw = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        except Exception as _e:
            print(f"[AI Numbers] error on ID {row['ID']}: {_e}")
            continue

        value = raw.strip().strip('".\'`').split()[0] if raw.strip() else ''
        if value.lower() in ('null', 'none', 'n/a', ''):
            continue

        mask_id = str(row['ID'])
        exact_mask_file = f"{current_image_name}_mask_layer_{mask_id}.png"
        row_mask = df_info['mask_file'] == exact_mask_file
        if not row_mask.any():
            exact_no_ext = f"{current_image_name}_mask_layer_{mask_id}"
            row_mask = df_info['mask_file'].apply(
                lambda x: str(x).replace('.png', '') == exact_no_ext)
        if row_mask.any():
            df_info.loc[row_mask, 'number'] = value
            updated += 1

    return updated


# --------------------------------------------------------------------------
# Figure-to-table metadata linkage
# --------------------------------------------------------------------------

_metadata_link_jobs = {}
_metadata_link_jobs_lock = threading.Lock()


def _resize_for_vision(image, max_side=3200):
    from PIL import Image as PILImage
    output = image.convert('RGB').copy()
    output.thumbnail((max_side, max_side), PILImage.Resampling.LANCZOS)
    return output


def _make_drawing_crop_sheet(image, cards):
    """Create labelled enlarged crops while retaining numbers below drawings."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    labels, tiles = [], []
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for index, card in enumerate(cards):
        label = alphabet[index] if index < 26 else f"Z{index}"
        x1, y1, x2, y2 = [int(v) for v in card['bbox']]
        width, height = image.size
        pad_x = max(50, int((x2 - x1) * .12))
        pad_top = max(35, int((y2 - y1) * .08))
        pad_bottom = max(130, int((y2 - y1) * .30))
        crop = image.crop((max(0, x1-pad_x), max(0, y1-pad_top),
                           min(width, x2+pad_x), min(height, y2+pad_bottom))).convert('RGB')
        crop.thumbnail((720, 430), PILImage.Resampling.LANCZOS)
        tile = PILImage.new('RGB', (740, 480), 'white')
        tile.paste(crop, ((740-crop.width)//2, 40))
        draw = ImageDraw.Draw(tile)
        draw.rectangle((0, 0, 740, 36), fill=(255, 80, 0))
        draw.text((12, 8), f"SOFTWARE LABEL {label}", fill='white', font=ImageFont.load_default())
        tiles.append(tile)
        labels.append((label, card['mask_file']))
    if not tiles:
        return PILImage.new('RGB', (10, 10), 'white'), labels
    columns = 2
    rows = (len(tiles) + columns - 1) // columns
    sheet = PILImage.new('RGB', (columns * 740, rows * 480), 'white')
    for index, tile in enumerate(tiles):
        sheet.paste(tile, ((index % columns) * 740, (index // columns) * 480))
    return sheet, labels


def _run_structured_vision(images, prompt, backend, api_key, model_name, max_tokens):
    if backend == 'openrouter':
        if not api_key:
            raise MetadataLinkError('OpenRouter API key is required')
        raw = call_openrouter_ai(images, prompt, api_key, model_name, max_tokens)
    else:
        model, processor = load_gemma_model()
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        inputs = processor(text=text, images=images, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[-1]
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        raw = processor.decode(outputs[0][input_len:], skip_special_tokens=True)
    return _parse_ai_json(raw)


class AppStructuredExtractor(StructuredExtractor):
    def __init__(self, backend='local', api_key='', model_name=''):
        self.backend = backend
        self.api_key = api_key
        self.model_name = model_name or 'google/gemini-2.5-flash'

    def extract_drawing_identifiers(self, image_path, cards, page_context):
        from PIL import Image as PILImage
        with PILImage.open(image_path) as source:
            source = source.convert('RGB')
            sheet, labels = _make_drawing_crop_sheet(source, cards)
            full_page = _resize_for_vision(source, 2800)
        label_text = ', '.join(label for label, _ in labels)
        prompt = (
            "Read this archaeological pottery figure. Image 1 is the full page and image 2 is a crop sheet. "
            "Orange labels are software labels, not publication numbers.\n"
            f"Return exactly one JSON object for software labels: {label_text}.\n"
            "Read the Figure caption, printed page number, and the small original vessel number below or near every drawing. "
            "Do not infer numbers from order. Use null when unreadable.\n"
            '{"figure_id":"2.1","figure_caption":"Figure 2.1 ...","printed_page":"19",'
            '"drawings":{"A":{"number":"1"},"B":{"number":"2"}}}'
        )
        result = _run_structured_vision([full_page, sheet], prompt, self.backend,
                                        self.api_key, self.model_name,
                                        min(4096, max(768, len(cards) * 45 + 384)))
        returned = result.get('drawings', {}) if isinstance(result, dict) else {}
        result['drawings'] = {mask_file: returned.get(label, {})
                              for label, mask_file in labels}
        return result

    def extract_table(self, image_path, crop, figure_id, expected_numbers, page_context):
        from PIL import Image as PILImage
        with PILImage.open(image_path) as source:
            source = source.convert('RGB')
            if crop:
                left, top, right, bottom = crop
                if bottom - top < 80:
                    return {"is_table": False, "rows": []}
                source = source.crop((left, top, right, bottom))
            table_image = _resize_for_vision(source, 3800)
        schema = ', '.join(f'"{column}"' for column in HESBAN_TABLE_COLUMNS)
        expected = ', '.join(expected_numbers) if expected_numbers else 'unknown'
        retry_note = "Return only the requested missing rows. " if page_context.get('retry_missing') else ""
        prompt = (
            "Extract a Hesban 11 pottery metadata table from this page image. " + retry_note +
            "The table may be below drawings, fill the page, continue from another page, or be absent. "
            "Preserve every symbol, abbreviation, punctuation mark, and multiline cell verbatim; join visual lines with newline characters. "
            "Never expand ** or invent ditto meanings. Use empty strings for blank cells.\n"
            f"Target figure: {figure_id}. Expected vessel numbers across all pages: {expected}.\n"
            f"Every row object must contain exactly these keys: {schema}.\n"
            '{"is_table":true,"figure_id":"2.1 or null","figure_caption":"raw caption or null",'
            '"printed_page":"20","rows":[{"table_no":"1"}]}. '
            "If there is no matching table, return {\"is_table\":false,\"rows\":[]}."
        )
        return _run_structured_vision([table_image], prompt, self.backend,
                                      self.api_key, self.model_name, 8192)


@app.route('/api/projects/<project_id>/metadata-link/run', methods=['POST'])
def run_project_metadata_link(project_id):
    project = project_manager.get_project(project_id)
    if not project:
        return jsonify({'error': 'Project not found', 'success': False}), 404
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object', 'success': False}), 400
    backend = str(data.get('backend') or data.get('ai_backend') or 'ocr').lower()
    if backend not in {'ocr', 'local', 'openrouter'}:
        return jsonify({'error': 'Backend must be ocr, local, or openrouter', 'success': False}), 400
    if backend == 'ocr':
        import importlib.util
        if not importlib.util.find_spec('paddleocr') or not importlib.util.find_spec('paddle'):
            return jsonify({
                'error': 'Local OCR is not installed. Run: pip install -r requirements-ocr.txt',
                'success': False, 'ocr_unavailable': True,
            }), 503
    with _metadata_link_jobs_lock:
        if _metadata_link_jobs.get(project_id):
            return jsonify({'error': 'A metadata-linking run is already active', 'success': False}), 409
        _metadata_link_jobs[project_id] = True
    project_path = project_manager.get_project_path(project_id)
    source_pdf = data.get('source_pdf') or None
    if source_pdf is not None:
        submitted_source = str(source_pdf)
        source_pdf = Path(submitted_source).name
        if source_pdf != submitted_source:
            with _metadata_link_jobs_lock:
                _metadata_link_jobs[project_id] = False
            return jsonify({'error': 'source_pdf must be a filename, not a path',
                            'success': False}), 400
        available_sources = {path.name for path in (project_path / 'pdf_source').glob('*.pdf')}
        if source_pdf not in available_sources:
            with _metadata_link_jobs_lock:
                _metadata_link_jobs[project_id] = False
            return jsonify({'error': 'Selected source PDF was not found', 'success': False}), 400
    api_key = str(data.get('openrouter_api_key', ''))
    model_name = str(data.get('openrouter_model', ''))

    def worker():
        try:
            extractor = (PaddleOCRStructuredExtractor() if backend == 'ocr' else
                         AppStructuredExtractor(backend=backend, api_key=api_key,
                                                model_name=model_name))
            linker = MetadataLinker(project_path, extractor, Hesban11Profile(), source_pdf)
            state = linker.run()
            project_manager.update_workflow_status(project_id, linkage_totals(state))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            state = load_linkage_state(project_path)
            state['status'] = 'error'
            state['error'] = str(exc)
            state['progress'] = {**state.get('progress', {}), 'message': str(exc)}
            save_linkage_state(project_path, state)
        finally:
            with _metadata_link_jobs_lock:
                _metadata_link_jobs[project_id] = False

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({'success': True, 'message': 'Metadata linking started'}), 202


@app.route('/api/projects/<project_id>/metadata-link/state', methods=['GET'])
def get_project_metadata_link_state(project_id):
    project_path = project_manager.get_project_path(project_id)
    if not project_path:
        return jsonify({'error': 'Project not found', 'success': False}), 404
    state = load_linkage_state(project_path)
    sources = [p.name for p in sorted((project_path / 'pdf_source').glob('*.pdf'))]
    import importlib.util
    ocr_available = bool(importlib.util.find_spec('paddleocr') and importlib.util.find_spec('paddle'))
    return jsonify({'success': True, 'state': state, 'sources': sources,
                    'ocr_available': ocr_available,
                    'active': bool(_metadata_link_jobs.get(project_id))})


@app.route('/api/projects/<project_id>/metadata-link/evidence/<path:filename>', methods=['GET'])
def get_project_metadata_link_evidence(project_id, filename):
    """Render review evidence with linked numbers or the stored table crop."""
    from PIL import Image as PILImage, ImageDraw, ImageFont
    project_path = project_manager.get_project_path(project_id)
    if not project_path:
        return jsonify({'error': 'Project not found', 'success': False}), 404
    image_path = project_path / 'images' / Path(filename).name
    if not image_path.exists():
        return jsonify({'error': 'Image not found', 'success': False}), 404
    state = load_linkage_state(project_path)
    requested_figure = str(request.args.get('figure', ''))
    target_figure = normalize_figure_id(requested_figure)
    figure = next((f for f in state.get('figures', [])
                   if str(f.get('figure_key', '')) == requested_figure or
                   normalize_figure_id(f.get('figure_id')) == target_figure), None)
    if not figure:
        return jsonify({'error': 'Figure not found', 'success': False}), 404
    image = PILImage.open(image_path).convert('RGB')
    if request.args.get('kind') == 'table':
        page = next((p for p in figure.get('table_pages', [])
                     if p.get('image_name') == Path(filename).name), None)
        boundary = page.get('boundary', {}) if page else {}
        if request.args.get('overlay') == '1' and boundary.get('table_bounds'):
            draw = ImageDraw.Draw(image)
            x1, y1, x2, y2 = [int(value) for value in boundary['table_bounds']]
            line_width = max(5, min(image.size) // 450)
            draw.rectangle((x1, y1, x2, y2), outline=(37, 99, 235), width=line_width)
            for key, color in (('upper_header_rule', (234, 179, 8)),
                               ('lower_header_rule', (34, 197, 94)),
                               ('closing_rule_y', (239, 68, 68))):
                value = boundary.get(key)
                if value is not None:
                    draw.line((x1, int(value), x2, int(value)), fill=color, width=line_width)
            columns = boundary.get('column_bounds', [])
            for index, column in enumerate(columns):
                if not isinstance(column, (list, tuple)) or len(column) != 2:
                    continue
                column_left, column_right = [int(value) for value in column]
                color = (14, 165, 233) if index % 2 == 0 else (168, 85, 247)
                draw.line((column_left, y1, column_left, y2), fill=color,
                          width=max(2, line_width // 2))
                if index == len(columns) - 1:
                    draw.line((column_right, y1, column_right, y2), fill=color,
                              width=max(2, line_width // 2))
            pad = max(20, round(image.width * .01))
            image = image.crop((max(0, x1-pad), max(0, y1-pad),
                                min(image.width, x2+pad), min(image.height, y2+pad)))
        elif request.args.get('ocr_row') and request.args.get('ocr_field'):
            diagnostic = next((
                item for item in (page or {}).get('ocr_diagnostics', [])
                if str(item.get('row', '')) == str(request.args.get('ocr_row', ''))
                and str(item.get('field', '')) == str(request.args.get('ocr_field', ''))
            ), None)
            if not diagnostic or len(diagnostic.get('crop', [])) != 4:
                return jsonify({'error': 'OCR diagnostic crop not found',
                                'success': False}), 404
            from ocr_extractor import _prepare_compact_cell
            image = _prepare_compact_cell(image.crop(tuple(
                int(value) for value in diagnostic['crop'])))
        elif page and page.get('crop'):
            image = image.crop(tuple(int(value) for value in page['crop']))
    else:
        draw = ImageDraw.Draw(image)
        highlighted_mask = Path(str(request.args.get('highlight', ''))).stem
        font_size = max(40, min(76, round(image.width / 52)))
        try:
            font = ImageFont.truetype("arialbd.ttf", font_size)
        except OSError:
            try:
                font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
            except OSError:
                font = ImageFont.load_default()
        line_width = max(4, min(image.size) // 500)
        measurement_line_width = max(2, line_width // 2)
        measurement_handle_radius = max(4, line_width)
        if request.args.get('measurement') == '1':
            calibration = figure.get('scale_calibrations', {}).get(Path(filename).name, {})
            if (calibration.get('p1') and calibration.get('p2') and
                    Path(str(calibration.get('evidence_image') or filename)).name == Path(filename).name):
                scale_color = ((22, 163, 74) if calibration.get('status') in
                               {'verified', 'verified_automatic', 'verified_manual'}
                               else (234, 179, 8))
                draw.line((*calibration['p1'], *calibration['p2']), fill=scale_color,
                          width=measurement_line_width)
                for point in (calibration['p1'], calibration['p2']):
                    radius = measurement_handle_radius
                    draw.ellipse((point[0] - radius, point[1] - radius,
                                  point[0] + radius, point[1] + radius),
                                 outline=scale_color, width=measurement_line_width)
        for drawing in figure.get('drawings', []):
            if drawing.get('image_name') != Path(filename).name:
                continue
            x1, y1, x2, y2 = [int(value) for value in drawing['bbox']]
            label = drawing.get('vessel_number') or '?'
            selected = highlighted_mask and Path(str(drawing.get('mask_file', ''))).stem == highlighted_mask
            box_color = (250, 204, 21) if selected else (255, 80, 0)
            draw.rectangle((x1, y1, x2, y2), outline=box_color,
                           width=line_width * 2 if selected else line_width)
            label_text = f"No. {label}"
            text_box = draw.textbbox((0, 0), label_text, font=font, stroke_width=1)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            pad = max(8, font_size // 5)
            tag_top = max(0, y1 - text_height - pad * 2)
            tag_right = min(image.width, x1 + text_width + pad * 2)
            draw.rectangle((x1, tag_top, tag_right, y1), fill=box_color)
            draw.text((x1 + pad, tag_top + pad // 2), label_text, fill='white',
                      font=font, stroke_width=1, stroke_fill=(120, 35, 0))
            if request.args.get('measurement') == '1':
                measurement = drawing.get('measurement', {})
                endpoints = measurement.get('rim_endpoints')
                measurement_color = ((34, 197, 94) if measurement.get('status') in
                                     {'verified', 'verified_automatic', 'verified_manual'}
                                     else (14, 165, 233))
                if endpoints and len(endpoints) == 2:
                    draw.line((*endpoints[0], *endpoints[1]), fill=measurement_color,
                              width=measurement_line_width)
                    for point in endpoints:
                        radius = measurement_handle_radius
                        draw.ellipse((point[0] - radius, point[1] - radius,
                                      point[0] + radius, point[1] + radius),
                                     outline=measurement_color,
                                     width=measurement_line_width)
                centreline = measurement.get('centreline')
                if centreline:
                    draw.line(tuple(centreline), fill=(168, 85, 247),
                              width=measurement_line_width)
    image.thumbnail((1800, 1800), PILImage.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    buffer.seek(0)
    return send_file(buffer, mimetype='image/png')


@app.route('/api/projects/<project_id>/metadata-link/figures/<path:figure_id>', methods=['PATCH'])
def update_project_metadata_link_figure(project_id, figure_id):
    project_path = project_manager.get_project_path(project_id)
    if not project_path:
        return jsonify({'error': 'Project not found', 'success': False}), 404
    state = load_linkage_state(project_path)
    target_id = normalize_figure_id(figure_id)
    figure = next((f for f in state.get('figures', [])
                   if str(f.get('figure_key', '')) == figure_id or
                   normalize_figure_id(f.get('figure_id')) == target_id), None)
    if not figure:
        return jsonify({'error': 'Figure not found', 'success': False}), 404
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object', 'success': False}), 400
    if (_metadata_link_jobs.get(project_id) and
            figure.get('processing_status') not in {'reviewable', 'ready', 'approved'}):
        return jsonify({'error': 'This figure is still being extracted',
                        'success': False, 'processing': True}), 409
    submitted_revision = data.get('reviewer_revision')
    current_revision = int(figure.get('reviewer_revision', 0) or 0)
    try:
        submitted_revision = (None if submitted_revision is None
                              else int(submitted_revision))
    except (TypeError, ValueError):
        return jsonify({'error': 'reviewer_revision must be a whole number',
                        'success': False}), 400
    if submitted_revision is not None and submitted_revision != current_revision:
        return jsonify({'error': 'This figure has a newer saved draft',
                        'success': False, 'conflict': True,
                        'figure': figure, 'reviewer_revision': current_revision}), 409
    changed_fields = []
    if 'figure_id' in data:
        corrected_id = get_profile(state.get('profile', 'hesban11')).normalize_figure(data['figure_id'])
        if not corrected_id:
            return jsonify({'error': 'figure_id cannot be empty', 'success': False}), 400
        collision = next((candidate for candidate in state.get('figures', [])
                          if candidate is not figure and
                          normalize_figure_id(candidate.get('figure_id')) == corrected_id), None)
        if collision:
            return jsonify({'error': f'Figure {corrected_id} already exists',
                            'success': False}), 409
        if corrected_id != normalize_figure_id(figure.get('figure_id')):
            figure['figure_id'] = corrected_id
            changed_fields.append('figure_id')
    if 'figure_caption' in data:
        figure['figure_caption'] = str(data['figure_caption'])
        changed_fields.append('figure_caption')
    if isinstance(data.get('drawing_numbers'), dict):
        for drawing in figure.get('drawings', []):
            if drawing.get('mask_file') in data['drawing_numbers']:
                value = data['drawing_numbers'][drawing['mask_file']]
                drawing['vessel_number'] = '' if value is None else str(value)
        changed_fields.append('drawing_numbers')
    scale_changed = False
    if isinstance(data.get('scale_calibrations'), dict) and data['scale_calibrations']:
        allowed_pages = {str(page.get('image_name', ''))
                         for page in figure.get('drawing_pages', [])}
        calibrations = dict(figure.get('scale_calibrations', {}))
        for image_name, submitted in data['scale_calibrations'].items():
            image_name = Path(str(image_name)).name
            if image_name not in allowed_pages or not isinstance(submitted, dict):
                return jsonify({'error': f'Invalid drawing-page calibration: {image_name}',
                                'success': False}), 400
            evidence_name = Path(str(submitted.get('evidence_image') or image_name)).name
            if evidence_name != image_name:
                manifest_pages = {page.get('image_name'): page for page in
                                  ensure_page_manifest(project_path).get('pages', [])}
                source_page = manifest_pages.get(image_name, {})
                evidence_page = manifest_pages.get(evidence_name, {})
                if (not evidence_page or evidence_page.get('source_pdf') != source_page.get('source_pdf') or
                        evidence_page.get('pdf_page_index') != source_page.get('pdf_page_index')):
                    return jsonify({'error': 'Scale evidence must come from the same PDF page',
                                    'success': False}), 400
            try:
                previous_calibration = calibrations.get(image_name, {})
                calibration = manual_calibration(
                    project_path / 'images' / evidence_name,
                    submitted.get('p1', []), submitted.get('p2', []),
                    float(submitted.get('real_cm', 10.0)))
            except (OSError, TypeError, ValueError) as exc:
                return jsonify({'error': str(exc), 'success': False}), 400
            if evidence_name != image_name:
                calibration['evidence_image'] = evidence_name
                calibration['applies_to_image'] = image_name
            calibration['reviewer_history'] = list(
                previous_calibration.get('reviewer_history', []))
            calibration['reviewer_history'].append({
                'action': 'scale_verified', 'at': calibration['verified_at'],
                'real_cm': calibration['real_cm'],
            })
            calibrations[image_name] = calibration
            for drawing in figure.get('drawings', []):
                if drawing.get('image_name') == image_name:
                    drawing['measurement'] = {
                        'status': 'unresolved', 'warning': 'scale_calibration_changed',
                        'drawing_fingerprint': drawing.get('fingerprint', ''),
                    }
            scale_changed = True
        figure['scale_calibrations'] = calibrations
        changed_fields.append('scale_calibrations')
    if scale_changed:
        measure_figure(project_path, figure, persist=False)
    if isinstance(data.get('measurements'), dict) and data['measurements']:
        drawings_by_mask = {str(drawing.get('mask_file')): drawing
                            for drawing in figure.get('drawings', [])}
        for mask_file, submitted in data['measurements'].items():
            drawing = drawings_by_mask.get(str(mask_file))
            if not drawing or not isinstance(submitted, dict):
                return jsonify({'error': f'Invalid drawing measurement: {mask_file}',
                                'success': False}), 400
            try:
                endpoints = submitted.get('rim_endpoints')
                if endpoints is not None:
                    from PIL import Image as PILImage
                    endpoints = [[float(value) for value in point] for point in endpoints]
                    if (len(endpoints) != 2 or any(len(point) != 2 for point in endpoints) or
                            any(not __import__('math').isfinite(value)
                                for point in endpoints for value in point)):
                        raise ValueError('Rim endpoints must contain two x/y points')
                    x1, y1, x2, y2 = [float(value) for value in drawing.get('bbox', [])]
                    with PILImage.open(project_path / 'images' /
                                       Path(str(drawing.get('image_name', ''))).name) as page_image:
                        page_width, page_height = page_image.size
                    margin_x, margin_y = max(10, (x2 - x1) * .08), max(10, (y2 - y1) * .08)
                    if any(point[0] < 0 or point[0] > page_width or
                           point[1] < 0 or point[1] > page_height or
                           point[0] < x1 - margin_x or point[0] > x2 + margin_x or
                           point[1] < y1 - margin_y or point[1] > y2 + margin_y
                           for point in endpoints):
                        raise ValueError('Rim endpoints must remain inside the drawing crop')
                calibration = figure.get('scale_calibrations', {}).get(
                    str(drawing.get('image_name', '')), {})
                value = submitted.get('verified_cm')
                value = None if value in (None, '') else float(value)
                drawing['measurement'] = verified_measurement(
                    drawing.get('measurement', {}), value_cm=value,
                    rim_endpoints=endpoints,
                    px_per_cm=float(calibration.get('px_per_cm', 0) or 0) or None)
                drawing['measurement']['drawing_fingerprint'] = drawing.get('fingerprint', '')
                drawing['measurement']['scale_page_fingerprint'] = calibration.get(
                    'page_fingerprint', '')
            except (TypeError, ValueError) as exc:
                return jsonify({'error': str(exc), 'success': False}), 400
        changed_fields.append('measurements')
    if 'table_rows' in data:
        if not isinstance(data['table_rows'], list) or not all(
                isinstance(row, dict) for row in data['table_rows']):
            return jsonify({'error': 'table_rows must be a list of objects', 'success': False}), 400
        figure['table_rows'] = [
            {column: '' if row.get(column) is None else str(row.get(column, ''))
             for column in HESBAN_TABLE_COLUMNS}
            for row in data['table_rows']
        ]
        changed_fields.append('table_rows')
    elif isinstance(data.get('row_operations'), list):
        rows = [dict(row) for row in figure.get('table_rows', [])]
        for operation in data['row_operations']:
            if not isinstance(operation, dict):
                return jsonify({'error': 'row_operations must contain objects',
                                'success': False}), 400
            action = operation.get('action')
            try:
                index = int(operation.get('index', -1))
            except (TypeError, ValueError):
                return jsonify({'error': 'Row operation index must be a whole number',
                                'success': False}), 400
            if action == 'delete' and 0 <= index < len(rows):
                rows.pop(index)
            elif action == 'duplicate' and 0 <= index < len(rows):
                rows.insert(index + 1, dict(rows[index]))
            elif action == 'add':
                submitted = operation.get('row', {})
                rows.append(submitted if isinstance(submitted, dict) else {})
            elif action == 'sort':
                rows.sort(key=lambda row: natural_key(str(row.get('table_no', ''))))
            else:
                return jsonify({'error': f'Invalid row operation: {action}',
                                'success': False}), 400
        figure['table_rows'] = [
            {column: '' if row.get(column) is None else str(row.get(column, ''))
             for column in HESBAN_TABLE_COLUMNS}
            for row in rows
        ]
        changed_fields.append('row_operations')
    if 'table_pages' in data:
        if not isinstance(data['table_pages'], list):
            return jsonify({'error': 'table_pages must be a list', 'success': False}), 400
        manifest_pages = {page.get('image_name'): page for page in
                          ensure_page_manifest(project_path).get('pages', [])}
        old_pages = {page.get('image_name'): page for page in figure.get('table_pages', [])}
        selected_pages = []
        drawing_sources = {str(page.get('source_pdf', ''))
                           for page in figure.get('drawing_pages', []) if page.get('source_pdf')}
        for submitted in data['table_pages']:
            name = submitted.get('image_name') if isinstance(submitted, dict) else submitted
            name = Path(str(name or '')).name
            page = manifest_pages.get(name)
            if not page:
                return jsonify({'error': f'Table page is not in the manifest: {name}',
                                'success': False}), 400
            if drawing_sources and page.get('source_pdf') not in drawing_sources:
                return jsonify({'error': 'A table page must come from the same source PDF as its drawings',
                                'success': False}), 400
            selected_pages.append({
                'image_name': name,
                'logical_index': page.get('logical_index'),
                'printed_page': page.get('printed_page', ''),
                'source_pdf': page.get('source_pdf', ''),
                'crop': old_pages.get(name, {}).get('crop'),
            })
        figure['table_pages'] = selected_pages
        changed_fields.append('table_pages')
    if data.get('review_status') in {'pending', 'rejected'}:
        figure['review_status'] = data['review_status']
        changed_fields.append('review_status')
    if isinstance(data.get('warning_overrides'), dict):
        # Validate against the warnings currently produced by this figure. A
        # client cannot turn an identity error into an overridable warning.
        validate_figure(figure, get_profile(state.get('profile', 'hesban11')))
        allowed = {warning.get('id'): warning for warning in figure.get('warnings', [])
                   if warning.get('overrideable')}
        previous_overrides = figure.get('warning_overrides', {})
        submitted_overrides = {}
        for warning_id, override in data['warning_overrides'].items():
            if warning_id not in allowed or not isinstance(override, dict):
                return jsonify({'error': f'Warning cannot be overridden: {warning_id}',
                                'success': False}), 400
            reason = str(override.get('reason', '')).strip()
            if not reason:
                return jsonify({'error': 'Every warning override requires a reason',
                                'success': False}), 400
            warning_code = str(allowed[warning_id].get('code', ''))
            if reason not in WARNING_OVERRIDE_REASONS.get(warning_code, set()):
                return jsonify({'error': f'Invalid review reason for {warning_code}',
                                'success': False}), 400
            previous = previous_overrides.get(warning_id, {})
            submitted_overrides[warning_id] = {
                'reason': reason,
                'note': str(override.get('note', '')).strip(),
                # The server owns this audit timestamp. Ordinary autosaves send
                # the active override again and must not rewrite when the
                # reviewer originally made the decision.
                'at': previous.get('at') or __import__('datetime').datetime.now(
                    __import__('datetime').timezone.utc).isoformat(),
            }
        figure['warning_overrides'] = submitted_overrides
        changed_fields.append('warning_overrides')
    if changed_fields:
        figure['reviewer_revision'] = current_revision + 1
        figure['draft_saved_at'] = __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat()
        if figure.get('review_status') == 'approved':
            figure['review_status'] = 'pending'
            figure['processing_status'] = 'reviewable'
            figure.pop('approved_at', None)
        figure.setdefault('review_history', []).append({
            'action': 'edited', 'at': __import__('datetime').datetime.now(
                __import__('datetime').timezone.utc).isoformat(),
            'fields': sorted(set(changed_fields)),
        })
    validate_figure(figure, get_profile(state.get('profile', 'hesban11')))
    try:
        save_linkage_state(project_path, state, expected_revisions={
            str(figure.get('figure_key', '')): current_revision,
        })
    except ReviewerRevisionConflict as conflict:
        latest = conflict.figure
        return jsonify({'error': str(conflict), 'success': False, 'conflict': True,
                        'figure': latest,
                        'reviewer_revision': int(latest.get('reviewer_revision', 0) or 0)}), 409
    if scale_changed or 'measurements' in changed_fields:
        persist_figure_measurements(project_path, figure)
    totals = linkage_totals(state)
    project_manager.update_workflow_status(project_id, totals)
    return jsonify({'success': True, 'figure': figure, 'totals': totals,
                    'reviewer_revision': figure.get('reviewer_revision', 0)})


@app.route('/api/projects/<project_id>/metadata-link/figures/<path:figure_id>/measure', methods=['POST'])
def measure_project_metadata_link_figure(project_id, figure_id):
    """Redetect the page ruler and rim-diameter suggestions for one figure."""
    project_path = project_manager.get_project_path(project_id)
    if not project_path:
        return jsonify({'error': 'Project not found', 'success': False}), 404
    state = load_linkage_state(project_path)
    target_id = normalize_figure_id(figure_id)
    figure = next((candidate for candidate in state.get('figures', [])
                   if str(candidate.get('figure_key', '')) == figure_id or
                   normalize_figure_id(candidate.get('figure_id')) == target_id), None)
    if not figure:
        return jsonify({'error': 'Figure not found', 'success': False}), 404
    if figure.get('processing_status') in {'processing', 'queued'}:
        return jsonify({'error': 'This figure is still being extracted or waiting to start',
                        'success': False}), 409
    data = request.get_json(silent=True) or {}
    current_revision = int(figure.get('reviewer_revision', 0) or 0)
    try:
        submitted_revision = int(data.get('reviewer_revision', current_revision))
    except (TypeError, ValueError):
        return jsonify({'error': 'reviewer_revision must be a whole number',
                        'success': False}), 400
    if submitted_revision != current_revision:
        return jsonify({'error': 'This figure has a newer saved draft',
                        'success': False, 'conflict': True, 'figure': figure,
                        'reviewer_revision': current_revision}), 409
    figure_sources = {page.get('source_pdf') for page in figure.get('drawing_pages', [])}
    figure_dpis = {page.get('render_dpi') for page in figure.get('drawing_pages', [])
                   if page.get('render_dpi') is not None}
    ratios = [calibration.get('px_per_cm')
              for other in state.get('figures', []) if other is not figure
              if ({page.get('source_pdf') for page in other.get('drawing_pages', [])} &
                  figure_sources)
              if ((not figure_dpis and not {page.get('render_dpi') for page in
                                             other.get('drawing_pages', [])
                                             if page.get('render_dpi') is not None}) or
                  {page.get('render_dpi') for page in other.get('drawing_pages', [])
                   if page.get('render_dpi') is not None} & figure_dpis)
              for calibration in other.get('scale_calibrations', {}).values()
              if (calibration.get('status') in {'suggested', 'verified', 'verified_automatic', 'verified_manual'} and
                  calibration.get('px_per_cm'))]
    try:
        # Preserve verified manual measurements; refresh automatic suggestions.
        for page_name, calibration in list(figure.get('scale_calibrations', {}).items()):
            if not (calibration.get('method') == 'manual' and
                    calibration.get('status') in {'verified', 'verified_manual'}):
                figure['scale_calibrations'].pop(page_name, None)
        measure_figure(project_path, figure,
                       project_ratios=ratios, persist=False)
        figure['reviewer_revision'] = current_revision + 1
        figure['draft_saved_at'] = __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc).isoformat()
        figure.setdefault('review_history', []).append({
            'action': 'measurements_redetected', 'at': figure['draft_saved_at'],
        })
        validate_figure(figure, get_profile(state.get('profile', 'hesban11')))
        save_linkage_state(project_path, state, expected_revisions={
            str(figure.get('figure_key', '')): current_revision,
        })
        persist_figure_measurements(project_path, figure)
    except ReviewerRevisionConflict as conflict:
        return jsonify({'error': str(conflict), 'success': False, 'conflict': True,
                        'figure': conflict.figure}), 409
    except Exception as exc:
        return jsonify({'error': str(exc), 'success': False}), 500
    return jsonify({'success': True, 'figure': figure,
                    'reviewer_revision': figure.get('reviewer_revision', 0)})


@app.route('/api/projects/<project_id>/metadata-link/figures/<path:figure_id>/rerun', methods=['POST'])
def rerun_project_metadata_link_figure(project_id, figure_id):
    """Rerun deterministic local OCR for one figure's assigned table pages."""
    if _metadata_link_jobs.get(project_id):
        return jsonify({'error': 'Wait for metadata linking to finish before rerunning OCR',
                        'success': False}), 409
    project_path = project_manager.get_project_path(project_id)
    if not project_path:
        return jsonify({'error': 'Project not found', 'success': False}), 404
    state = load_linkage_state(project_path)
    target_id = normalize_figure_id(figure_id)
    figure = next((candidate for candidate in state.get('figures', [])
                   if str(candidate.get('figure_key', '')) == figure_id or
                   normalize_figure_id(candidate.get('figure_id')) == target_id), None)
    if not figure:
        return jsonify({'error': 'Figure not found', 'success': False}), 404
    if not figure.get('table_pages'):
        return jsonify({'error': 'Assign at least one table page before rerunning OCR',
                        'success': False}), 409
    import importlib.util
    if not importlib.util.find_spec('paddleocr') or not importlib.util.find_spec('paddle'):
        return jsonify({'error': 'Local OCR is not installed. Run: pip install -r requirements-ocr.txt',
                        'success': False, 'ocr_unavailable': True}), 503

    with _metadata_link_jobs_lock:
        if _metadata_link_jobs.get(project_id):
            return jsonify({'error': 'Metadata linking is already running for this project',
                            'success': False}), 409
        _metadata_link_jobs[project_id] = True
    try:
        # Persist this before the synchronous OCR work starts. Other Flask
        # requests can now display the correct state and must not edit the same
        # figure while its draft is being replaced by fresh extraction output.
        figure['processing_status'] = 'processing'
        save_linkage_state(project_path, state)
        profile = get_profile(state.get('profile', 'hesban11'))
        manifest = ensure_page_manifest(project_path, profile.slug)
        manifest_pages = {page.get('image_name'): page for page in manifest.get('pages', [])}
        expected = sorted({profile.normalize_number(drawing.get('vessel_number'))
                           for drawing in figure.get('drawings', [])
                           if drawing.get('vessel_number')}, key=natural_key)
        extractor = PaddleOCRStructuredExtractor()
        rows = []
        warnings = []
        updated_pages = []
        seen = set()
        for assigned in figure.get('table_pages', []):
            page = manifest_pages.get(assigned.get('image_name'))
            if not page:
                warnings.append({'code': 'page_not_in_manifest',
                                 'message': f"Table page {assigned.get('image_name')} is not in the manifest."})
                updated_pages.append(assigned)
                continue
            context = profile.detect_figure_context(page.get('page_text', ''))
            context.update(page)
            context['figure_id'] = figure.get('figure_id', '')
            context['figure_caption'] = figure.get('figure_caption', '')
            table = extractor.extract_table(
                project_path / 'images' / page['image_name'],
                tuple(assigned['crop']) if assigned.get('crop') else None,
                figure.get('figure_id', ''), expected, context) or {}
            boundary = table.get('boundary', {}) if isinstance(table.get('boundary'), dict) else {}
            updated = {**assigned, 'logical_index': page.get('logical_index'),
                       'printed_page': table.get('printed_page') or page.get('printed_page', ''),
                       'source_pdf': page.get('source_pdf', ''), 'boundary': boundary,
                       'ocr_diagnostics': [
                           item for item in table.get('ocr_diagnostics', [])
                           if isinstance(item, dict)
                       ]}
            updated_pages.append(updated)
            if not table.get('is_table'):
                warnings.extend(warning for warning in table.get('warnings', [])
                                if isinstance(warning, dict))
                continue
            for row in table.get('rows', []):
                clean = {column: '' if row.get(column) is None else str(row.get(column, ''))
                         for column in profile.table_columns}
                number = profile.normalize_number(clean.get('table_no'))
                key = (number, page['image_name'])
                if not number or key in seen:
                    continue
                seen.add(key)
                clean['normalized_table_no'] = number
                clean['source_image'] = page['image_name']
                clean['source_printed_page'] = updated['printed_page']
                rows.append(clean)
            warnings.extend(warning for warning in table.get('warnings', [])
                            if isinstance(warning, dict))

        for index, page in enumerate(updated_pages):
            boundary = page.get('boundary', {})
            if not boundary or boundary.get('has_closing_rule'):
                if boundary:
                    boundary['continues'] = False
                continue
            following = updated_pages[index+1] if index+1 < len(updated_pages) else None
            adjacent = bool(following and int(following.get('logical_index', -2)) ==
                            int(page.get('logical_index', -1)) + 1)
            if adjacent and following.get('boundary', {}).get('header_confirmed'):
                boundary['continues'] = True
            else:
                warnings.append({'code': 'missing_table_end',
                                 'message': f"Table page {page.get('image_name')} has no closing rule or verified continuation."})

        figure['table_pages'] = updated_pages
        figure['table_rows'] = rows
        figure['extraction_warnings'] = warnings
        figure['review_status'] = 'pending'
        figure.setdefault('review_history', []).append({
            'action': 'ocr_rerun',
            'at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            'pages': [page.get('image_name') for page in updated_pages],
        })
        validate_figure(figure, profile)
        figure['processing_status'] = 'reviewable'
        save_linkage_state(project_path, state)
        totals = linkage_totals(state)
        project_manager.update_workflow_status(project_id, totals)
        return jsonify({'success': True, 'figure': figure, 'totals': totals})
    except MetadataLinkError as exc:
        return jsonify({'error': str(exc), 'success': False}), 409
    finally:
        try:
            latest_state = load_linkage_state(project_path)
            latest_figure = next((candidate for candidate in latest_state.get('figures', [])
                                  if str(candidate.get('figure_key', '')) ==
                                  str(figure.get('figure_key', ''))), None)
            if latest_figure and latest_figure.get('processing_status') == 'processing':
                latest_figure['processing_status'] = 'reviewable'
                save_linkage_state(project_path, latest_state)
        except Exception:
            app.logger.exception('Could not restore figure review state after OCR rerun')
        with _metadata_link_jobs_lock:
            _metadata_link_jobs[project_id] = False


@app.route('/api/projects/<project_id>/metadata-link/apply', methods=['POST'])
def apply_project_metadata_links(project_id):
    project_path = project_manager.get_project_path(project_id)
    if not project_path:
        return jsonify({'error': 'Project not found', 'success': False}), 404
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify({'error': 'Request body must be a JSON object', 'success': False}), 400
    state = load_linkage_state(project_path)
    figure_ids = data.get('figure_ids')
    if figure_ids is not None and not isinstance(figure_ids, list):
        return jsonify({'error': 'figure_ids must be a list', 'success': False}), 400
    figure_ids = figure_ids or [f.get('figure_id') for f in state.get('figures', [])
                                if f.get('status') == 'ready']
    try:
        result = apply_approved_figures(
            project_path, figure_ids, bool(data.get('replace_imported', False)))
        project_manager.update_workflow_status(project_id, result['totals'])
        return jsonify({'success': True, **result})
    except MetadataLinkError as exc:
        return jsonify({'error': str(exc), 'success': False}), 409


@app.route('/api/projects/<project_id>/tabular/export', methods=['POST'])
def export_project_tabular_csv(project_id):
    """Save combined tabular CSV (mask_info.csv) in the project folder"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404

        cards_path = project_manager.get_project_path(project_id, 'cards')
        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'Cards folder not found', 'success': False}), 404

        # Get source CSV from temp location (if exists)
        temp_csv_path = cards_path / 'mask_info.csv'
        
        # Save to project root
        project_path = project_manager.get_project_path(project_id)
        export_csv_path = project_path / f"{project_id}_mask_info.csv"
        
        if temp_csv_path.exists():
            import shutil
            shutil.copy2(temp_csv_path, export_csv_path)
        else:
            return jsonify({'error': 'Combined CSV not found. Please save tabular data first.', 'success': False}), 404

        return jsonify({
            'message': f'CSV exported to {export_csv_path.name}',
            'path': str(export_csv_path),
            'success': True
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== POST PROCESSING ====================

@app.route('/api/postprocess/process', methods=['POST'])
def process_folder():
    """Process folder with classification model"""
    try:
        data = request.json
        folder = data.get('folder')
        flip_vertical = data.get('flip_vertical', True)
        flip_horizontal = data.get('flip_horizontal', True)
        
        if not folder:
            return jsonify({'error': 'Folder is required', 'success': False}), 400
        
        # Set flip options
        second_step_processor.set_flip_options(flip_vertical, flip_horizontal)
        
        # Process
        results = second_step_processor.process_folder(folder)
        
        if results.empty:
            return jsonify({'error': 'No images were processed', 'success': False}), 400
        
        return jsonify({
            'message': f'Successfully processed {len(results)} images',
            'count': len(results),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/postprocess/load', methods=['POST'])
def load_processed_image():
    """Load processed image for review"""
    try:
        from PIL import Image

        data = request.json
        folder = data.get('folder')
        img_num = int(data.get('img_num', 0))
        
        if not folder:
            return jsonify({'error': 'Folder is required', 'success': False}), 400
        
        results = second_step_processor.load_results(folder)
        if results.empty or img_num >= len(results):
            return jsonify({'error': 'No results found', 'success': False}), 404
        
        row = results.iloc[img_num]
        
        # Load images
        import io
        
        original_path = second_step_processor.get_original_path(folder, row['filename'])
        transformed_path = second_step_processor.get_transformed_path(folder, row['filename'])
        
        def img_to_base64(path):
            if path.exists():
                img = Image.open(path)
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                return base64.b64encode(buffer.getvalue()).decode()
            return None
        
        original_b64 = img_to_base64(original_path)
        transformed_b64 = img_to_base64(transformed_path)
        
        return jsonify({
            'original': f'data:image/png;base64,{original_b64}' if original_b64 else None,
            'transformed': f'data:image/png;base64,{transformed_b64}' if transformed_b64 else None,
            'type': row['type'],
            'position': row['position'],
            'rotation': row['rotation'],
            'current': img_num,
            'total': len(results) - 1,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/postprocess/flip', methods=['POST'])
def flip_image():
    """Manually flip an image"""
    try:
        data = request.json
        folder = data.get('folder')
        img_num = int(data.get('img_num', 0))
        flip_type = data.get('flip_type')  # 'vertical' or 'horizontal'
        
        if not all([folder, flip_type]):
            return jsonify({'error': 'Missing required data', 'success': False}), 400
        
        results = second_step_processor.load_results(folder)
        if results.empty or img_num >= len(results):
            return jsonify({'error': 'Image not found', 'success': False}), 404
        
        filename = results.iloc[img_num]['filename']
        
        # Flip
        flipped = second_step_processor.manual_flip(folder, filename, flip_type)
        
        if flipped is None:
            return jsonify({'error': 'Failed to flip image', 'success': False}), 500
        
        # Return updated image
        import io
        
        buffer = io.BytesIO()
        flipped.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'image': f'data:image/png;base64,{img_base64}',
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/postprocess/update-type', methods=['POST'])
def update_type():
    """Update type classification"""
    try:
        data = request.json
        folder = data.get('folder')
        img_num = int(data.get('img_num', 0))
        new_type = data.get('type')
        
        if not all([folder, new_type]):
            return jsonify({'error': 'Missing required data', 'success': False}), 400
        
        results = second_step_processor.load_results(folder)
        if results.empty or img_num >= len(results):
            return jsonify({'error': 'Image not found', 'success': False}), 404
        
        filename = results.iloc[img_num]['filename']
        second_step_processor.update_result(folder, filename, {'type': new_type})
        
        return jsonify({
            'message': f'Updated type to {new_type}',
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/postprocess/merge', methods=['POST'])
def merge_annotations():
    """Merge annotations with classifications"""
    try:
        data = request.json
        folder = data.get('folder')
        
        if not folder:
            return jsonify({'error': 'Folder is required', 'success': False}), 400
        
        # Get paths
        annots_path = PRED_OUTPUT_DIR / folder / "mask_info.csv"
        transformed_folder = second_step_processor.get_transformed_folder_path(folder)
        results_path = transformed_folder / "classifications.csv"
        
        if not annots_path.exists():
            return jsonify({'error': 'Annotations file not found', 'success': False}), 404
        if not results_path.exists():
            return jsonify({'error': 'Classifications not found. Process images first.', 'success': False}), 404
        
        # Load and merge
        annots_df = pd.read_csv(annots_path)
        results_df = pd.read_csv(results_path)
        
        annots_df.rename(columns={'mask_file': 'filename'}, inplace=True)
        results_df['filename'] = results_df['filename'].str.replace('.png', '')
        
        merged_df = pd.merge(
            annots_df,
            results_df[['filename', 'type']],
            on='filename',
            how='left'
        )
        
        if 'file' in merged_df.columns:
            merged_df = merged_df.drop('file', axis=1)
        
        # Save
        output_path = transformed_folder / "merged_annotations.csv"
        merged_df.to_csv(output_path, index=False)
        
        return jsonify({
            'message': 'Successfully merged annotations with classifications',
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== EXPORT ====================

@app.route('/api/export', methods=['POST'])
def export_results():
    """Export final results"""
    try:
        data = request.json
        folder = data.get('folder')
        acronym = data.get('acronym')
        export_pdf = data.get('export_pdf', False)
        page_size = data.get('page_size', 'A4')
        scale_factor = float(data.get('scale_factor', 1.0))
        
        if not all([folder, acronym]):
            return jsonify({'error': 'Folder and acronym are required', 'success': False}), 400
        
        # Validate acronym
        if not acronym.replace('_', '').isalnum():
            return jsonify({'error': 'Acronym can only contain letters, numbers, and underscores', 'success': False}), 400
        
        result = export_processor.export_results(
            folder=folder,
            acronym=acronym,
            export_pdf=export_pdf,
            page_size=page_size,
            scale_factor=scale_factor
        )
        
        return jsonify({
            'message': result,
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== STATIC FILES ====================

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


# ==================== PROJECT-AWARE POSTPROCESSING ENDPOINTS ====================

@app.route('/api/projects/<project_id>/postprocess', methods=['POST'])
def process_project_cards(project_id):
    """Process all cards in a project with classification model"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        data = request.json
        flip_vertical = data.get('flip_vertical', True)
        flip_horizontal = data.get('flip_horizontal', True)
        
        # Get project paths
        cards_path = project_manager.get_project_path(project_id, 'cards')
        cards_modified_path = project_manager.get_project_path(project_id, 'cards_modified')
        
        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'No cards folder found in project', 'success': False}), 404
        
        # Create cards_modified folder
        cards_modified_path.mkdir(exist_ok=True)
        
        # Set flip options
        second_step_processor.set_flip_options(flip_vertical, flip_horizontal)
        
        def _natural_key(s):
            return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', str(s.name))]

        # Get all card images
        card_files = sorted([f for f in cards_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']],
                           key=_natural_key)
        
        if not card_files:
            return jsonify({'error': 'No card images found', 'success': False}), 404
        
        total_cards = len(card_files)
        
        # Initialize progress
        update_operation_progress('postprocess', 0, total_cards, 'Starting post-processing...')
        
        # Process each card
        results = []
        for idx, card_file in enumerate(card_files):
            try:
                # Update progress
                update_operation_progress('postprocess', idx + 1, total_cards, 
                                        f'Processing image {idx + 1} of {total_cards}')
                
                # Process the image
                type_pred, pos_pred, rot_pred, transformed_image = second_step_processor.process_image(str(card_file))
                
                if all((type_pred, pos_pred, rot_pred)) and transformed_image:
                    # Save transformed image to cards_modified
                    transformed_path = cards_modified_path / card_file.name
                    transformed_image.save(transformed_path)
                    
                    results.append({
                        'filename': card_file.name,
                        'type': type_pred,
                        'position': pos_pred,
                        'rotation': rot_pred
                    })
                    print(f"Processed {card_file.name}: Type={type_pred}, Pos={pos_pred}, Rot={rot_pred}")
                    
            except Exception as e:
                print(f"Error processing {card_file.name}: {e}")
                continue
        
        # Clear progress
        clear_operation_progress()
        
        # Save classifications
        if results:
            import pandas as pd
            results_df = pd.DataFrame(results)
            classifications_path = cards_modified_path / 'classifications.csv'
            results_df.to_csv(classifications_path, index=False)
            print(f"Saved classifications for {len(results)} cards")
        
        return jsonify({
            'message': f'Successfully processed {len(results)} cards',
            'count': len(results),
            'success': True
        })
        
    except Exception as e:
        clear_operation_progress()
        print(f"Error processing project cards: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/postprocess/flip', methods=['POST'])
def flip_project_card(project_id):
    """Flip a single card image"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        data = request.json
        img_num = int(data.get('img_num', 0))
        card_filename = str(data.get('card_filename', '')).strip()
        flip_type = data.get('flip_type', 'vertical')

        # Get project paths
        cards_path = project_manager.get_project_path(project_id, 'cards')
        cards_modified_path = project_manager.get_project_path(project_id, 'cards_modified')

        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'No cards folder found', 'success': False}), 404

        # Resolve the target card. Prefer the explicit filename (robust); fall
        # back to a NATURALLY-sorted index so it matches the /cards listing order
        # (plain sort put _10 before _2 and flipped the wrong card).
        card_file = None
        if card_filename:
            candidate = cards_path / os.path.basename(card_filename)
            if candidate.exists():
                card_file = candidate
        if card_file is None:
            def _natural_key(f):
                return [int(c) if c.isdigit() else c.lower()
                        for c in re.split(r'(\d+)', f.name)]
            card_files = sorted([f for f in cards_path.iterdir()
                                 if f.suffix.lower() in ['.png', '.jpg', '.jpeg']],
                                key=_natural_key)
            if img_num < 0 or img_num >= len(card_files):
                return jsonify({'error': 'Invalid image number', 'success': False}), 400
            card_file = card_files[img_num]
        
        # Check if a modified version already exists, use that instead
        modified_file = cards_modified_path / card_file.name
        if modified_file.exists():
            source_file = modified_file
        else:
            source_file = card_file
        
        # Load image from the appropriate source
        from PIL import Image
        img = Image.open(source_file)
        
        # Apply flip
        if flip_type == 'vertical':
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        elif flip_type == 'horizontal':
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        # Save to cards_modified (always save the result here)
        cards_modified_path.mkdir(parents=True, exist_ok=True)
        output_path = cards_modified_path / card_file.name
        img.save(output_path)
        
        # Convert to base64 for display
        import io
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return jsonify({
            'success': True,
            'image': f'data:image/png;base64,{img_base64}'
        })
        
    except Exception as e:
        print(f"Error flipping card: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# --- Excluded cards (cards the operator removes from the final export) -------

def _excluded_cards_path(cards_modified_dir):
    return Path(cards_modified_dir) / 'excluded_cards.json'


def _read_excluded_cards(cards_modified_dir):
    """Return the list of excluded card filenames (empty if none)."""
    path = _excluded_cards_path(cards_modified_dir)
    if not path.exists():
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get('excluded', [])
    except Exception as e:
        print(f"Error reading excluded_cards.json: {e}")
        return []


def _write_excluded_cards(cards_modified_dir, filenames):
    cards_modified_dir = Path(cards_modified_dir)
    cards_modified_dir.mkdir(parents=True, exist_ok=True)
    with open(_excluded_cards_path(cards_modified_dir), 'w') as f:
        json.dump(sorted(set(filenames)), f)


@app.route('/api/projects/<project_id>/postprocess/exclude', methods=['POST'])
def exclude_project_card(project_id):
    """Mark a card as excluded/included from the final export."""
    try:
        if not project_manager.get_project(project_id):
            return jsonify({'error': 'Project not found', 'success': False}), 404

        data = request.json or {}
        filename = os.path.basename(str(data.get('filename', '')).strip())
        excluded = bool(data.get('excluded', True))
        if not filename:
            return jsonify({'error': 'Filename is required', 'success': False}), 400

        cards_path = project_manager.get_project_path(project_id, 'cards')
        if not cards_path:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        cards_modified_dir = cards_path.parent / 'cards_modified'

        current = set(_read_excluded_cards(cards_modified_dir))
        if excluded:
            current.add(filename)
        else:
            current.discard(filename)
        _write_excluded_cards(cards_modified_dir, current)

        return jsonify({'success': True, 'excluded': excluded, 'count': len(current)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/postprocess/update-type', methods=['POST'])
def update_project_card_type(project_id):
    """Update the type classification for a card"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        data = request.json
        filename = data.get('filename', '')
        new_type = data.get('type', '')
        
        if not filename:
            return jsonify({'error': 'Filename is required', 'success': False}), 400
        
        # Get project cards folder
        cards_path = project_manager.get_project_path(project_id, 'cards')
        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'No cards folder found', 'success': False}), 404
        
        # Try cards_modified first (where classifications.csv is usually saved after processing)
        project_root = cards_path.parent
        cards_modified_path = project_root / 'cards_modified'
        classifications_csv = cards_modified_path / 'classifications.csv'
        
        if not classifications_csv.exists():
            # Fallback to cards folder
            classifications_csv = cards_path / 'classifications.csv'
        
        if not classifications_csv.exists():
            return jsonify({'error': 'Classifications file not found. Run processing first.', 'success': False}), 404
        
        # Load classifications CSV
        df = pd.read_csv(classifications_csv)
        
        # Find the row with matching filename
        mask = df['filename'] == filename
        if not mask.any():
            return jsonify({'error': f'File {filename} not found in classifications', 'success': False}), 404
        
        # Update type
        df.loc[mask, 'type'] = new_type
        
        # Save back to CSV
        df.to_csv(classifications_csv, index=False)
        
        print(f"Updated type for {filename} to {new_type} in {classifications_csv}")
        
        return jsonify({
            'success': True,
            'message': 'Type updated successfully'
        })
        
    except Exception as e:
        print(f"Error updating type: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/postprocess/merge', methods=['POST'])
def merge_project_annotations(project_id):
    """Merge mask annotations with classifications"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        # Get project cards folder
        cards_path = project_manager.get_project_path(project_id, 'cards')
        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'No cards folder found', 'success': False}), 404
        
        # Check for required CSV files
        mask_info_csv = cards_path / 'mask_info.csv'
        classifications_csv = cards_path / 'classifications.csv'
        
        if not mask_info_csv.exists():
            return jsonify({'error': 'mask_info.csv not found. Extract cards first.', 'success': False}), 404
        
        if not classifications_csv.exists():
            return jsonify({'error': 'classifications.csv not found. Run processing first.', 'success': False}), 404
        
        # Load both CSVs
        df_mask = pd.read_csv(mask_info_csv)
        df_class = pd.read_csv(classifications_csv)
        
        # Merge on filename (mask_file matches image in classifications)
        merged = pd.merge(
            df_mask, 
            df_class[['image', 'type', 'is_correct']], 
            left_on='mask_file', 
            right_on='image', 
            how='left'
        )
        
        # Save merged annotations
        merged_csv = cards_path / 'merged_annotations.csv'
        merged.to_csv(merged_csv, index=False)
        
        return jsonify({
            'success': True,
            'message': f'Successfully merged {len(merged)} annotations'
        })
        
    except Exception as e:
        print(f"Error merging annotations: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/projects/<project_id>/export/legacy', methods=['POST'])
def legacy_export_project_results(project_id):
    """Export final results for a project (with auto-merge if CSV exists)"""
    try:
        # Verify project exists
        project_metadata = project_manager.get_project(project_id)
        if not project_metadata:
            return jsonify({'error': 'Project not found', 'success': False}), 404
        
        data = request.json
        acronym = data.get('acronym')
        
        if not acronym:
            return jsonify({'error': 'Acronym is required', 'success': False}), 400
        
        # Validate acronym
        if not acronym.replace('_', '').isalnum():
            return jsonify({'error': 'Acronym can only contain letters, numbers, and underscores', 'success': False}), 400
        
        # Get project paths
        cards_path = project_manager.get_project_path(project_id, 'cards')
        cards_modified_path = project_manager.get_project_path(project_id, 'cards_modified')
        if not cards_path or not cards_path.exists():
            return jsonify({'error': 'No cards folder found', 'success': False}), 404
        
        # Auto-merge: read tabular data directly from cards/mask_info.csv (always up to date)
        combined_csv_path = cards_path / 'mask_info.csv'
        if combined_csv_path.exists():
            print("Found combined CSV, merging with classifications...")
            try:
                # Merge the CSVs
                import pandas as pd
                
                # Load combined CSV (tabular data)
                combined_df = pd.read_csv(combined_csv_path, dtype=str, keep_default_na=False)
                combined_df = migrate_linkage_columns(combined_df, drop_legacy=True)
                print(f"Loaded combined CSV with {len(combined_df)} rows")
                print(f"Combined CSV columns: {list(combined_df.columns)}")
                
                # Load classifications if they exist
                classifications_path = cards_modified_path / 'classifications.csv' if cards_modified_path else None
                if classifications_path and classifications_path.exists():
                    classifications_df = pd.read_csv(classifications_path)
                    print(f"Loaded classifications CSV with {len(classifications_df)} rows")
                    print(f"Classifications CSV columns: {list(classifications_df.columns)}")
                    
                    # Ensure filename columns are compatible (remove .png if present in one but not other)
                    if 'filename' in combined_df.columns and 'filename' in classifications_df.columns:
                        # Normalize filenames - remove extension for matching
                        combined_df['filename_base'] = combined_df['filename'].str.replace('.png', '').str.replace('.jpg', '')
                        classifications_df['filename_base'] = classifications_df['filename'].str.replace('.png', '').str.replace('.jpg', '')
                        
                        # Merge on normalized filename
                        merged = pd.merge(
                            combined_df,
                            classifications_df,
                            on='filename_base',
                            how='left',
                            suffixes=('', '_class')
                        )
                        
                        # Keep original filename from combined
                        if 'filename_class' in merged.columns:
                            merged = merged.drop('filename_class', axis=1)
                        merged = merged.drop('filename_base', axis=1)
                        
                        # Save merged annotations
                        cards_modified_path.mkdir(exist_ok=True)
                        merged_path = cards_modified_path / 'merged_annotations.csv'
                        merged.to_csv(merged_path, index=False)
                        print(f"Auto-merged {len(merged)} annotations to {merged_path}")
                        print(f"Merged columns: {list(merged.columns)}")
                    else:
                        print("Warning: 'filename' column not found in one of the CSVs")
                        merged_path = cards_modified_path / 'merged_annotations.csv'
                        combined_df.to_csv(merged_path, index=False)
                else:
                    # No classifications, use combined CSV directly as merged
                    cards_modified_path.mkdir(exist_ok=True)
                    merged_path = cards_modified_path / 'merged_annotations.csv'
                    combined_df.to_csv(merged_path, index=False)
                    print("No classifications found, using combined CSV as merged")
                    
            except Exception as e:
                print(f"Warning: Auto-merge failed: {e}")
                import traceback
                traceback.print_exc()
                # Continue with export anyway
        
        # Prefer post-processed cards only when that folder actually contains
        # card images. Metadata/classification CSVs alone must not hide the
        # original cards and produce an empty export.
        modified_has_images = bool(
            cards_modified_path and cards_modified_path.exists() and any(
                path.is_file() and path.suffix.lower() in {'.png', '.jpg', '.jpeg'}
                for path in cards_modified_path.iterdir()))
        if modified_has_images:
            export_folder = cards_modified_path
        else:
            export_folder = cards_path
        
        # Load merged annotations if available
        import pandas as pd
        merged_path = cards_modified_path / 'merged_annotations.csv' if cards_modified_path else None
        
        # Create final metadata with new IDs
        metadata_df = None
        if merged_path and merged_path.exists():
            metadata_df = pd.read_csv(merged_path)
            print(f"Loaded merged annotations: {len(metadata_df)} rows")
            print(f"Merged columns: {list(metadata_df.columns)}")
        elif combined_csv_path.exists():
            # Use combined CSV if no merged exists
            metadata_df = pd.read_csv(combined_csv_path)
            print(f"Loaded combined CSV: {len(metadata_df)} rows")
            print(f"Combined columns: {list(metadata_df.columns)}")
        
        # Load classifications to ensure we have type column
        classifications_df = None
        classifications_path = cards_modified_path / 'classifications.csv' if cards_modified_path else None
        if classifications_path and classifications_path.exists():
            classifications_df = pd.read_csv(classifications_path)
            print(f"Loaded classifications: {len(classifications_df)} rows")
            print(f"Classifications columns: {list(classifications_df.columns)}")
        
        # Create ZIP in temporary location
        import tempfile
        import zipfile
        
        temp_dir = tempfile.mkdtemp()
        zip_path = Path(temp_dir) / f"{acronym}.zip"
        
        try:
            # Get all card images sorted
            # Skip cards the operator excluded in Post Processing
            excluded_set = set(_read_excluded_cards(cards_modified_path)) if cards_modified_path else set()
            card_images = sorted([f for f in export_folder.iterdir()
                                  if f.suffix.lower() in ['.png', '.jpg', '.jpeg']
                                  and f.name not in excluded_set])
            print(f"Found {len(card_images)} card images to export ({len(excluded_set)} excluded)")
            
            # Prepare final metadata with new IDs
            final_metadata = []
            
            for idx, img_file in enumerate(card_images, 1):
                new_id_with_ext = f"{acronym}_{idx}{img_file.suffix}"  # Include extension
                
                # Initialize row with id
                row_data = {'id': new_id_with_ext}
                
                # Try to find matching row in metadata
                if metadata_df is not None:
                    # Try different column names for matching
                    for col in ['mask_file', 'filename', 'Filename', 'file']:
                        if col in metadata_df.columns:
                            # Normalize both sides for comparison (remove extensions)
                            img_base = img_file.stem  # filename without extension
                            
                            # Try exact match first
                            mask = metadata_df[col] == img_file.name
                            if not mask.any():
                                # Try without extension
                                mask = metadata_df[col].str.replace('.png', '').str.replace('.jpg', '').str.replace('.jpeg', '') == img_base
                            
                            if mask.any():
                                row = metadata_df[mask].iloc[0]
                                
                                # Copy all columns except unwanted ones
                                exclude_cols = ['mask_file', 'filename', 'Filename', 'filename_base', 'file', 'ID', 'id']
                                for metadata_col in metadata_df.columns:
                                    if metadata_col not in exclude_cols:
                                        row_data[metadata_col] = row[metadata_col]
                                
                                print(f"Matched {img_file.name} via column '{col}'")
                                break
                
                # Ensure 'type' is present from classifications if available
                if classifications_df is not None and 'type' not in row_data:
                    # Try to match with classifications
                    img_base = img_file.stem
                    for col in ['filename', 'Filename']:
                        if col in classifications_df.columns:
                            mask = classifications_df[col].str.replace('.png', '').str.replace('.jpg', '').str.replace('.jpeg', '') == img_base
                            if mask.any():
                                class_row = classifications_df[mask].iloc[0]
                                if 'type' in class_row:
                                    row_data['type'] = class_row['type']
                                    print(f"Added type '{class_row['type']}' for {img_file.name}")
                                break
                
                final_metadata.append(row_data)
            
            # Create final metadata DataFrame
            final_df = pd.DataFrame(final_metadata)
            
            # Reorder columns: id first, then type (if present), then others alphabetically
            cols = ['id']
            if 'type' in final_df.columns:
                cols.append('type')
            # Add remaining columns alphabetically
            remaining = sorted([col for col in final_df.columns if col not in cols])
            cols.extend(remaining)
            final_df = final_df[cols]
            
            # Save metadata to temp file — all fields as text
            metadata_temp_path = Path(temp_dir) / f"{acronym}_metadata.csv"
            final_df = final_df.astype(str).replace({'nan': '', 'None': ''})
            final_df.to_csv(metadata_temp_path, index=False)
            print(f"Created final metadata with {len(final_df)} rows and columns: {list(final_df.columns)}")
            
            # Create ZIP
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add images with new names
                for idx, img_file in enumerate(card_images, 1):
                    new_name = f"{acronym}_{idx}{img_file.suffix}"
                    zipf.write(img_file, new_name)
                    print(f"Added {img_file.name} as {new_name}")
                
                # Add metadata
                zipf.write(metadata_temp_path, f"{acronym}_metadata.csv")
                print("Added metadata CSV")
        
            # Send the ZIP file
            return send_file(
                str(zip_path),
                as_attachment=True,
                download_name=f"{acronym}.zip",
                mimetype='application/zip'
            )
            
        finally:
            # Cleanup will happen after send_file completes
            pass
        
    except Exception as e:
        print(f"Error exporting project results: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e), 'success': False}), 500


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found', 'success': False}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error', 'success': False}), 500


@app.route('/api/projects/<project_id>/thumbnail/<filename>')
def serve_project_thumbnail(project_id, filename):
    """Serve a thumbnail version of a project image"""
    try:
        from PIL import Image
        import io
        
        # Get images path
        images_path = project_manager.get_project_path(project_id, 'images')
        if not images_path or not images_path.exists():
            return jsonify({'error': 'Images folder not found', 'success': False}), 404
        
        image_path = images_path / filename
        if not image_path.exists():
            return jsonify({'error': 'Image not found', 'success': False}), 404
        
        # Open and create thumbnail
        with Image.open(image_path) as img:
            # Convert to RGB if necessary (for JPEG compatibility)
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Create thumbnail (max 300px on longest side, maintain aspect ratio)
            img.thumbnail((300, 300), Image.Resampling.LANCZOS)
            
            # Save to memory buffer
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85, optimize=True)
            buffer.seek(0)
            
            return send_file(buffer, mimetype='image/jpeg')
            
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


if __name__ == '__main__':
    print("\n" + "="*80)
    print(" 🏺 SherdScope Flask Application 🔍")
    print("="*80)
    print("\n🚀 Starting server...")
    print("📝 Browser will open at: http://localhost:5001")
    print("💡 Initialization will continue in background...")
    print("\n" + "="*80 + "\n")
    
    # Open browser immediately after Flask starts
    import webbrowser
    import threading
    
    def open_browser():
        import time
        time.sleep(1)  # Wait 1 second for Flask to start
        webbrowser.open('http://localhost:5001')
        print("🌐 Browser opened!")
    
    # Start browser in separate thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True,
        threaded=True,
        use_reloader=False  # Disable reloader to prevent double initialization
    )
