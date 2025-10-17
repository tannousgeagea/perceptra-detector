import os

# Base directory
BASE_DIR = "./"

# Folder structure
FOLDERS = [
    "tests/fixtures",
    "examples",
    "configs",
    "perceptra_detector/core",
    "perceptra_detector/backends",
    "perceptra_detector/utils",
    "perceptra_detector/api",
    "perceptra_detector/client",
    "perceptra_detector/cli",
    "perceptra_detector/training",
]

# File structure (empty files)
FILES = [
    "README.md",
    "LICENSE",
    "setup.py",
    "pyproject.toml",
    "requirements.txt",
    "requirements-dev.txt",
    "Dockerfile",
    "docker-compose.yml",
    ".env.example",
    ".gitignore",
    # Tests
    "tests/__init__.py",
    "tests/test_detector.py",
    "tests/test_backends.py",
    "tests/test_api.py",
    # Examples
    "examples/basic_inference.py",
    "examples/video_processing.py",
    "examples/custom_model.py",
    "examples/batch_processing.py",
    # Configs
    "configs/default.yaml",
    "configs/production.yaml",
    # Core
    "perceptra_detector/__init__.py",
    "perceptra_detector/__version__.py",
    "perceptra_detector/core/__init__.py",
    "perceptra_detector/core/detector.py",
    "perceptra_detector/core/base.py",
    "perceptra_detector/core/registry.py",
    "perceptra_detector/core/schemas.py",
    # Backends
    "perceptra_detector/backends/__init__.py",
    "perceptra_detector/backends/yolo.py",
    "perceptra_detector/backends/detr.py",
    "perceptra_detector/backends/rt_detr.py",
    "perceptra_detector/backends/custom.py",
    # Utils
    "perceptra_detector/utils/__init__.py",
    "perceptra_detector/utils/image.py",
    "perceptra_detector/utils/video.py",
    "perceptra_detector/utils/visualization.py",
    "perceptra_detector/utils/config.py",
    "perceptra_detector/utils/logging.py",
    # API
    "perceptra_detector/api/__init__.py",
    "perceptra_detector/api/server.py",
    "perceptra_detector/api/routes.py",
    "perceptra_detector/api/models.py",
    "perceptra_detector/api/middleware.py",
    # Client
    "perceptra_detector/client/__init__.py",
    "perceptra_detector/client/sdk.py",
    # CLI
    "perceptra_detector/cli/__init__.py",
    "perceptra_detector/cli/commands.py",
    # Training
    "perceptra_detector/training/__init__.py",
    "perceptra_detector/training/README.md",
]

def create_project_structure():
    """Create folders and empty files for the perceptra-detector project."""
    os.makedirs(BASE_DIR, exist_ok=True)
    print(f"📁 Base directory: {BASE_DIR}")

    # Create folders
    for folder in FOLDERS:
        path = os.path.join(BASE_DIR, folder)
        os.makedirs(path, exist_ok=True)
        print(f"  📂 Created folder: {path}")

    # Create files (skip if they exist)
    for file in FILES:
        path = os.path.join(BASE_DIR, file)
        if os.path.exists(path):
            print(f"  ⚠️  Skipped (exists): {path}")
            continue
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w", encoding="utf-8").close()
        print(f"  📝 Created file: {path}")

    print("\n✅ Project structure created successfully (empty files, skipped existing).")

if __name__ == "__main__":
    create_project_structure()
