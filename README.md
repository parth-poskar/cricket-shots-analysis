# 🏏 AI-Powered Cricket Shot Analysis System

A comprehensive Python-based cricket shot analysis platform that leverages **computer vision**, **pose estimation**, and **machine learning** to analyze and evaluate cricket shots in detail. The system processes video files, detects human pose landmarks using MediaPipe, computes biomechanical metrics, classifies shot types using deep learning, and predicts shot quality using an ensemble of supervised ML models.

**Key Features:**
- 🎬 Real-time video processing with pose estimation
- 📊 Biomechanical metrics extraction (angles, balance, footwork)
- 🤖 Multi-model ML ensemble for shot quality prediction
- 🎯 Shot-type classification (Cover Drive, Pull Shot, Straight Drive, etc.)
- 📈 Detailed per-aspect evaluation and feedback
- 🌐 FastAPI backend with REST endpoints
- 🎨 Real-time skeleton overlay and metric visualization

---

## 📋 Table of Contents

1. [Project Architecture](#project-architecture)
2. [Installation & Setup](#installation--setup)
3. [Technical Pipeline](#technical-pipeline)
4. [Models & Algorithms](#models--algorithms)
5. [Module Documentation](#module-documentation)
6. [Dataset Schema](#dataset-schema)
7. [UML Diagram](#uml-diagram)
8. [Quick Start Guide](#quick-start-guide)
9. [API Documentation](#api-documentation)

---

## 🏗️ Project Architecture

### Directory Structure

```
cricket_cover_drive_analysis/
│
├── 📄 main.py                          # Main video analysis pipeline entry point
├── 📄 train_model.py                   # ML model training orchestrator
├── 📄 train_shot_model.py              # ResNet18 shot type classifier trainer
├── 📄 train_shot_landmark_model.py     # Landmark-based shot classifier trainer
│
├── 🔧 src/                             # Core processing modules
│   ├── config.py                       # Configuration, thresholds, and constants
│   ├── video_processing.py             # Video I/O, codec handling, streaming
│   ├── pose_estimation.py              # MediaPipe Pose wrapper and interface
│   ├── biomechanical_metrics.py        # Feature extraction engine
│   ├── overlay_utils.py                # Visualization (skeleton, metrics)
│   ├── evaluation.py                   # Rule-based per-aspect scoring
│   ├── shot_classifier.py              # ResNet18-based shot type detector
│   ├── shot_landmark_classifier.py     # Pose-landmark-based shot detector
│   ├── frame_extractor.py              # Frame sampling and preprocessing
│   ├── ml_model.py                     # ⭐ ML training & inference engine
│   └── dataset_utils.py                # Dataset generation, loading, export
│
├── 🌐 backend/
│   └── app.py                          # FastAPI REST API server
│
├── 🤖 models/                          # Trained models directory
│   ├── best_model.pkl                  # Best ML ensemble (shot quality)
│   ├── shot_type_model.pth             # ResNet18 shot classifier (PyTorch)
│   ├── shot_landmark_model.pkl         # Landmark classifier (scikit-learn)
│   ├── shot_type_classes.json          # Shot type class mapping
│   └── shot_landmark_classes.json      # Landmark classifier class mapping
│
├── 📊 data/                            # Datasets and training data
│   ├── shot_quality_dataset.csv        # Quality classification dataset (900 samples)
│   ├── shot_landmark_dataset.csv       # Landmark-based features (multi-shot types)
│   └── shot_dataset/                   # Image folders for shot classification
│       ├── 0. Cut Shot/
│       ├── 1. Cover Drive/
│       ├── 2. Straight Drive/
│       └── 3. Pull Shot/
│
├── 📁 output/                          # Analysis results
│   ├── {video}_result.json             # Comprehensive analysis results
│   ├── evaluation.json                 # Per-aspect scores
│   ├── ml_prediction.json              # ML model predictions
│   └── {video}_annotated.mp4           # Output video with overlays
│
├── 📦 frontend/                        # React/Next.js web interface
├── 🧪 test_folder/                     # Batch processing input directory
└── 📝 requirements.txt                 # Python dependencies
```

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager
- 4GB+ RAM (8GB+ recommended)
- Optional: GPU (CUDA 11.8+) for faster processing

### Step 1: Clone and Navigate

```bash
cd cricket_cover_drive_analysis
```

### Step 2: Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux (Bash):**
```bash
python -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **opencv-python** | Latest | Video I/O and frame processing |
| **mediapipe** | 0.10.11 | Pose estimation (33 landmarks) |
| **numpy** | Latest | Numerical computations |
| **pandas** | Latest | Dataset handling and CSV I/O |
| **scikit-learn** | Latest | SVM, Random Forest, Gradient Boosting |
| **xgboost** | Latest | XGBoost classifier |
| **torch** | Latest | PyTorch for ResNet18 |
| **torchvision** | Latest | Pre-trained model weights |
| **joblib** | Latest | Model serialization (.pkl) |
| **fastapi** | Latest | REST API framework |
| **uvicorn** | Latest | ASGI server |

---

## 🔄 Technical Pipeline

### Complete Workflow Diagram

```
INPUT VIDEO
    ↓
[Video Processing Module]
    ├─ Extract frames (30 fps downsampled)
    ├─ Normalize frame dimensions
    └─ Handle codec compatibility
    ↓
[Pose Estimation Pipeline] ← MediaPipe Pose
    ├─ Detect 33 body landmarks per frame
    ├─ Confidence filtering (min_confidence=0.5)
    └─ Landmark coordinates (x, y, visibility)
    ↓
[Biomechanical Feature Extraction]
    ├─ Elbow Angle (shoulder-elbow-wrist)
    ├─ Spine Lean (hip-shoulder-vertical)
    ├─ Head Over Knee (head-knee distance)
    ├─ Foot Direction (ankle-foot-index)
    └─ Additional metrics (wrist height, hip rotation, etc.)
    ↓
[Shot Type Classification] ─→ (Optional) Two Approaches:
    ├─ ResNet18 Image Classifier (PyTorch)
    └─ Landmark-Based Classifier (scikit-learn)
    ↓
[Shot Quality ML Ensemble]
    ├─ SVM Classifier
    ├─ Random Forest Classifier
    ├─ Gradient Boosting Classifier
    ├─ XGBoost Classifier
    └─ Voting Ensemble (combines all 4)
    ↓
[Per-Aspect Rule-Based Evaluation]
    ├─ Footwork Score
    ├─ Head Position Score
    ├─ Swing Control Score
    ├─ Balance Score
    └─ Follow-through Score
    ↓
[Visualization & Output]
    ├─ Skeleton overlay on video
    ├─ Real-time metric display
    ├─ Annotated output video (.mp4)
    └─ JSON result files
    ↓
OUTPUT FILES
├─ {video}_result.json (complete analysis)
├─ evaluation.json (per-aspect scores)
├─ ml_prediction.json (quality prediction)
└─ {video}_annotated.mp4 (annotated video)
```

---

## 🤖 Models & Algorithms

### 1. **Shot Quality Prediction Ensemble**

#### Model Architecture
Four base models trained with 5-fold cross-validation, combined via soft voting:

```
Input Features (4 dimensions)
    ├─ elbow_angle:     Angle at dominant elbow (degrees) [0-180]
    ├─ spine_lean:      Spine angle from vertical (degrees) [0-180]
    ├─ head_over_knee:  Normalized head-to-knee distance [0-1]
    └─ foot_direction:  Foot alignment angle (degrees) [0-180]
         ↓
    [Feature Normalization - StandardScaler]
         ↓
    ├─ SVM (C=1.0, kernel='rbf')
    ├─ Random Forest (n_estimators=100, depth=10)
    ├─ Gradient Boosting (n_estimators=100, lr=0.1)
    └─ XGBoost (n_estimators=100, depth=6, learning_rate=0.1)
         ↓
    [Soft Voting Ensemble - Average Probabilities]
         ↓
Output Classes: ["Poor", "Average", "Good"]
    with probability distribution
```

#### Performance Metrics (Typical)

```
Model                  CV Accuracy ± Std    Test Accuracy    Best?
─────────────────────────────────────────────────────────────────
SVM                         0.943 ± 0.011        0.940          
Random Forest               0.958 ± 0.009        0.953          
Gradient Boosting           0.961 ± 0.008        0.957          
XGBoost                     0.966 ± 0.007        0.963          
🏆 Ensemble (Voting)        0.971 ± 0.006        0.967        ✓
```

#### Mathematical Formulation

**Standard Scaler Normalization:**
$$X_{scaled} = \frac{X - \mu}{\sigma}$$

**SVM Decision Function:**
$$f(x) = \sum_{i=1}^{n} \alpha_i y_i K(x_i, x) + b$$
where $K$ is RBF kernel: $K(x, x') = \exp(-\gamma \|x - x'\|^2)$

**Random Forest:**
$$\hat{f}(x) = \frac{1}{B} \sum_{b=1}^{B} T_b(x)$$
where $T_b$ is the $b$-th decision tree

**Gradient Boosting:**
$$F_M(x) = \sum_{m=1}^{M} \eta \cdot f_m(x)$$
where $f_m$ is weak learner and $\eta$ is learning rate

**Voting Ensemble (Soft):**
$$P(y|x) = \frac{1}{M} \sum_{m=1}^{M} P_m(y|x)$$
averaging class probabilities from all M models

---

### 2. **Shot Type Classification (ResNet18)**

#### Architecture

```
Input Image (224×224×3 RGB)
    ↓
[ResNet18 Pre-trained Backbone]
├─ Conv1 (7×7, 64 filters)
├─ Layer1 (2 BasicBlocks, 64 channels)
├─ Layer2 (2 BasicBlocks, 128 channels) [stride=2]
├─ Layer3 (2 BasicBlocks, 256 channels) [stride=2]  ← Frozen
├─ Layer4 (2 BasicBlocks, 512 channels) [stride=2]  ← Fine-tuned
    ↓
[Global Average Pooling]
    ↓
[Fully Connected Layer]
  Input: 512 features → Output: num_classes
    ↓
[Softmax] → Output: Class Probabilities
```

#### Training Configuration

- **Pre-training:** ImageNet weights (ResNet18_Weights.IMAGENET1K_V1)
- **Fine-tuning Strategy:** Freeze layers except `layer4` and `fc` (transfer learning)
- **Data Augmentation:**
  - Random Horizontal Flip (p=0.5)
  - Random Rotation (±15°)
  - ColorJitter (brightness, contrast, saturation, hue)
  - Normalization (mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
  
- **Training Hyperparameters:**
  - Optimizer: Adam
  - Learning Rate: 3×10⁻⁴ with ReduceLROnPlateau scheduler
  - Batch Size: 32
  - Epochs: 20 (with early stopping on validation accuracy)
  - Train/Val Split: 80/20
  - Loss: CrossEntropyLoss

#### Classes Detected
- Cover Drive
- Straight Drive
- Pull Shot
- Cut Shot
- Leg Glance
- (Extensible based on dataset)

---

### 3. **Shot Type Classification (Landmark-Based)**

#### Feature Set (8 biomechanical features)

```python
features = {
    "elbow_angle": float,              # Degrees [0-180]
    "wrist_height": float,             # Normalized [0-1]
    "hip_rotation": float,             # Degrees [0-180]
    "knee_bend": float,                # Degrees [0-180]
    "spine_lean": float,               # Degrees [0-180]
    "foot_stance_width": float,        # Normalized [0-2]
    "bat_plane": float,                # Degrees [0-90]
    "weight_forward": float            # Normalized [0-1]
}
```

#### Algorithm
- **Base Model:** scikit-learn classifier (SVM, Random Forest, or Gradient Boosting)
- **Input:** 8-dimensional feature vector
- **Output:** Shot type with confidence score
- **Advantage:** Interpretable, domain-driven features vs. pixel-level

---

## 📚 Module Documentation

### Core Modules

#### **config.py** — Configuration & Constants

```python
# Video Processing
MIN_DETECTION_CONFIDENCE = 0.5    # MediaPipe confidence threshold
MIN_TRACKING_CONFIDENCE = 0.5     # Tracking confidence threshold

# Biomechanical Thresholds (for rule-based evaluation)
ELBOW_ANGLE_GOOD_THRESHOLD = (90, 180)      # Range for good swing
HEAD_OVER_KNEE_GOOD_THRESHOLD = 0.1         # Head stability distance
FOOT_DIRECTION_GOOD_RANGE = (80, 100)       # Foot alignment
SPINE_LEAN_GOOD_RANGE = (80, 100)           # Spine angle range

# Phase Detection Thresholds
HIP_VELOCITY_THRESHOLD = 0.02                # Hip movement detection
ARM_VELOCITY_THRESHOLD = 0.03                # Arm movement detection
IMPACT_WRIST_VELOCITY_THRESHOLD = 0.05       # Ball contact detection

# Skill Grading
ADVANCED_SCORE = 8                           # Advanced skill threshold
INTERMEDIATE_SCORE = 5                       # Intermediate threshold

OUTPUT_DIR = "output"                        # Results directory
```

---

#### **pose_estimation.py** — MediaPipe Wrapper

```python
class PoseEstimator:
    """
    Encapsulates MediaPipe Pose detection with error handling.
    
    Methods:
    --------
    __init__()
        Initialize MediaPipe Pose model
        
    process_frame(frame: np.ndarray) → (frame, results)
        Process single video frame
        Returns BGR frame and pose estimation results
        
    get_landmarks(results) → np.ndarray
        Extract 33 body landmarks from results
        Returns normalized coordinates (x, y, visibility) per landmark
    """
    
    # 33 Pose Landmarks Tracked:
    # 0-10: Head and face (nose, eyes, ears)
    # 11-16: Arms (shoulders, elbows, wrists)
    # 17-28: Torso and legs (hips, knees, ankles)
    # 29-32: Foot indices (heel, foot, toe)
```

**MediaPipe Pose Landmarks Index:**
```
      0 - Nose
  11      12 - Shoulders
  13      14 - Elbows
  15      16 - Wrists
  23      24 - Hips
  25      26 - Knees
  27      28 - Ankles
  29-32   - Foot indices
```

---

#### **biomechanical_metrics.py** — Feature Extraction Engine

```python
def calculate_angle(point_a, point_b, point_c) → float:
    """
    Calculate angle at point_b formed by points a, b, c.
    
    Formula:
        angle = arctan2(c.y - b.y, c.x - b.x) - arctan2(a.y - b.y, a.x - b.x)
        angle_degrees = |angle| × 180 / π
        if angle > 180°: angle = 360° - angle
    
    Parameters:
        point_a, point_b, point_c: [x, y] coordinates
    
    Returns:
        Angle in degrees [0, 180]
    """

def calculate_metrics(landmarks, mp_pose) → dict:
    """
    Extract 4 primary biomechanical features from pose landmarks.
    
    Metrics:
    --------
    1. elbow_angle
       - Calculated from: shoulder → elbow → wrist
       - Range: [0°, 180°]
       - Good swing: 90-155° (high elbow technique)
    
    2. spine_lean
       - Calculated from: hip → shoulder → vertical
       - Range: [0°, 180°]
       - Good posture: 80-100° (upright with slight lean forward)
    
    3. head_over_knee
       - Distance: |head.x - knee.x|
       - Range: [0, 1] (normalized)
       - Good balance: < 0.1 (head directly over front knee)
    
    4. foot_direction
       - Angle from ankle through ankle to foot index
       - Range: [0°, 180°]
       - Good alignment: 80-100° (front foot points towards target)
    
    Returns:
        {
            "elbow_angle": float,
            "spine_lean": float,
            "head_over_knee": float,
            "foot_direction": float,
            "wrist": [x, y],
            "shoulder": [x, y]
        }
    """
```

---

#### **ml_model.py** — ML Training & Inference

```python
class MLTrainer:
    """
    Trains and compares four ML models with cross-validation.
    
    Models Trained:
    ───────────────
    1. SVM (Support Vector Machine)
       - Kernel: RBF (Radial Basis Function)
       - C: 1.0 (regularization parameter)
       - Scales features via StandardScaler pipeline
    
    2. Random Forest
       - Estimators: 100 trees
       - Max depth: 10
       - Random state: 42
    
    3. Gradient Boosting
       - Estimators: 100
       - Learning rate: 0.1
       - Subsample: 1.0
    
    4. XGBoost
       - Estimators: 100
       - Max depth: 6
       - Learning rate: 0.1
    
    5. Voting Ensemble (Meta-model)
       - Combines all 4 models
       - Voting method: soft (averaged probabilities)
    
    Methods:
    --------
    train_all(X: np.ndarray, y: np.ndarray) → None
        Train all models with stratified 5-fold CV
        X: shape (n_samples, 4) - biomechanical features
        y: shape (n_samples,) - class labels [0, 1, 2]
    
    evaluate(X_test, y_test) → dict
        Compute test accuracy and classification report
    
    save_best(path: str) → None
        Save best model to disk via joblib
    
    print_comparison() → None
        Display CV and test accuracy table
    """

class MLPredictor:
    """
    Loads a saved model and performs inference.
    
    Methods:
    --------
    predict_from_metrics(metrics_dict: dict) → dict
        Input biomechanical metrics dict
        Returns:
        {
            "label": "Good",  # or "Poor", "Average"
            "confidence": 0.95,
            "probabilities": {
                "Poor": 0.02,
                "Average": 0.03,
                "Good": 0.95
            }
        }
    """
```

---

#### **dataset_utils.py** — Dataset Management

```python
def generate_synthetic_dataset(
    n_samples: int = 900,
    seed: int = 42,
    output_path: str = "data/shot_quality_dataset.csv",
    noise_scale: float = 1.5
) → pd.DataFrame:
    """
    Generate synthetic labeled dataset using biomechanical domain knowledge.
    
    Dataset Schema:
    ───────────────
    - video_id: unique identifier
    - frame_index: frame number
    - elbow_angle: [0, 180] degrees
    - spine_lean: [0, 180] degrees
    - head_over_knee: [0, 1] normalized
    - foot_direction: [0, 180] degrees
    - label: "Poor" | "Average" | "Good"
    
    Class Distribution (equal split):
    ─────────────────────────────────
    Good (300):    Close to ideal biomechanical ranges
    Average (300): Boundary/partially correct metrics
    Poor (300):    Clearly outside optimal ranges
    
    Parameters:
    -----------
    n_samples: total samples (divided equally among 3 classes)
    seed: random seed for reproducibility
    output_path: CSV save location
    noise_scale: Gaussian noise std for feature diversity
    
    Returns:
    --------
    pd.DataFrame with columns: video_id, frame_index,
                                elbow_angle, spine_lean,
                                head_over_knee, foot_direction, label
    """

def load_dataset(path: str) → (np.ndarray, np.ndarray):
    """
    Load CSV dataset and return features and labels.
    
    Returns:
    --------
    X: np.ndarray shape (n_samples, 4) - features
    y: np.ndarray shape (n_samples,) - encoded labels [0, 1, 2]
    """

def export_features_from_video(video_path: str) → pd.DataFrame:
    """
    Extract raw biomechanical features from a video.
    Output can be manually labeled by coach for real training data.
    """
```

---

#### **shot_classifier.py** — ResNet18 Shot Classifier

```python
class ShotClassifier:
    """
    ResNet18-based image classifier for cricket shot type detection.
    
    Architecture:
    ──────────────
    Input (224×224×3) → ResNet18 backbone → FC layer → Softmax
    
    Methods:
    --------
    predict(frame: np.ndarray) → dict
        Input: Single video frame (BGR)
        Returns:
        {
            "shot_type": "Cover Drive",
            "confidence": 0.89,
            "all_probabilities": {
                "Cover Drive": 0.89,
                "Pull Shot": 0.07,
                "Cut Shot": 0.04,
                ...
            }
        }
    """
```

---

#### **evaluation.py** — Rule-Based Per-Aspect Scoring

```python
def save_evaluation(
    footwork_scores: list,
    head_position_scores: list,
    swing_control_scores: list,
    balance_scores: list,
    follow_through_scores: list,
    reference_feedback: str = None
) → None:
    """
    Calculate per-aspect scores and save to evaluation.json
    
    Aspects (Each scored 0-10):
    ────────────────────────────
    1. Footwork
       - Front foot alignment
       - Stance width
       - Movement patterns
       Feedback: "Ensure your front foot points towards the cover region."
    
    2. Head Position
       - Head stability
       - Head over front knee distance
       - No head movement during swing
       Feedback: "Keep your head still and over your front knee for balance."
    
    3. Swing Control
       - Elbow height
       - Bat path smoothness
       - Follow-through control
       Feedback: "High elbow is key. Ensure a full swing with control."
    
    4. Balance
       - Spine angle
       - Weight distribution
       - Base stability
       Feedback: "Maintain stable base and good spine angle throughout."
    
    5. Follow-through
       - Swing completion
       - Bat trajectory
       - Body rotation
       Feedback: "Complete your swing with high, full follow-through."
    
    Output JSON:
    {
        "Footwork": {
            "score": 7.5,
            "feedback": "..."
        },
        "Head Position": {
            "score": 8.2,
            "feedback": "..."
        },
        ...
    }
    """
```

---

#### **backend/app.py** — FastAPI Server

```python
app = FastAPI(title="Cricket Analytics API")

# ─────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────

@app.get("/health")
def health() → dict:
    """
    Health check endpoint.
    
    Returns:
    --------
    {
        "status": "ok",
        "ml_model": "ready" | "missing",
        "shot_model": "ready" | "missing"
    }
    """

@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)) → dict:
    """
    Upload and analyze video.
    
    Input:
    ──────
    Multipart form-data with video file
    
    Processing:
    ───────────
    1. Save to temporary file
    2. Run main.analyze_video() pipeline
    3. Extract results from JSON
    
    Returns:
    --------
    {
        "video_filename": "myshot_annotated.mp4",
        "quality_prediction": {
            "label": "Good",
            "confidence": 0.95,
            "probabilities": {...}
        },
        "shot_type": "Cover Drive",
        "evaluation": {
            "Footwork": {"score": 7.5, "feedback": "..."},
            ...
        },
        "metrics_per_frame": [
            {"frame": 0, "elbow_angle": 120, ...},
            ...
        ]
    }
    """

@app.get("/video/{video_stem}")
async def stream_video(video_stem: str) → FileResponse:
    """Stream annotated video file."""

@app.get("/results/{video_stem}")
async def get_results(video_stem: str) → dict:
    """Retrieve analysis results from _result.json"""
```

---

## 📊 Dataset Schema

### 1. Shot Quality Dataset (`shot_quality_dataset.csv`)

**Purpose:** Training data for ML shot quality classification

**Schema:**
```csv
video_id,frame_index,elbow_angle,spine_lean,head_over_knee,foot_direction,label
synthetic_good_0070,70,121.22,87.99,0.0304,95.50,Good
synthetic_poor_0227,227,53.98,46.55,0.2712,51.27,Poor
synthetic_avg_0288,288,91.61,75.92,0.1149,82.10,Average
```

**Columns:**
| Column | Type | Range | Meaning |
|--------|------|-------|---------|
| `video_id` | String | - | Unique video identifier |
| `frame_index` | Integer | [0, ∞) | Frame number in video |
| `elbow_angle` | Float | [0, 180] | Angle at elbow (degrees) |
| `spine_lean` | Float | [0, 180] | Spine angle from vertical (degrees) |
| `head_over_knee` | Float | [0, 1] | Normalized head-to-knee distance |
| `foot_direction` | Float | [0, 180] | Front foot alignment (degrees) |
| `label` | Category | {Poor, Average, Good} | Shot quality class |

**Dataset Statistics:**
- Total samples: 900
- Classes: 3 (Poor: 300, Average: 300, Good: 300)
- Feature dimensions: 4
- Train/Test split: 80/20
- Cross-validation: 5-fold stratified

---

### 2. Shot Landmark Dataset (`shot_landmark_dataset.csv`)

**Purpose:** Training data for landmark-based shot type classification

**Schema:**
```csv
shot_type,elbow_angle,wrist_height,hip_rotation,knee_bend,spine_lean,foot_stance_width,bat_plane,weight_forward
Pull Shot,122.85,0.75,56.21,90.95,21.37,1.46,12.97,0.21
Cover Drive,142.47,0.68,23.80,148.29,22.98,0.81,37.78,0.51
```

**Columns:**
| Column | Type | Range | Meaning |
|--------|------|-------|---------|
| `shot_type` | Category | {Cover Drive, Straight Drive, Pull Shot, Cut Shot, Leg Glance} | Cricket shot type |
| `elbow_angle` | Float | [0, 180] | Elbow angle (degrees) |
| `wrist_height` | Float | [0, 1] | Normalized wrist height |
| `hip_rotation` | Float | [0, 180] | Hip rotation angle (degrees) |
| `knee_bend` | Float | [0, 180] | Front knee bend angle (degrees) |
| `spine_lean` | Float | [0, 180] | Spine lean angle (degrees) |
| `foot_stance_width` | Float | [0, 2] | Normalized feet separation |
| `bat_plane` | Float | [0, 90] | Bat swing plane angle (0=horizontal, 90=vertical) |
| `weight_forward` | Float | [0, 1] | Weight distribution (0=back foot, 1=front foot) |

---

### 3. Shot Classification Dataset (`data/shot_dataset/`)

**Purpose:** Image dataset for ResNet18 shot type classifier

**Structure:**
```
shot_dataset/
├── 0. Cut Shot/              # ~200 images of cut shots
├── 1. Cover Drive/           # ~200 images of cover drives
├── 2. Straight Drive/        # ~200 images of straight drives
├── 3. Pull Shot/             # ~200 images of pull shots
├── 4. Leg Glance/            # ~200 images of leg glances
└── ...                       # More shot types as needed
```

**Image Specifications:**
- Format: JPG/PNG
- Size: Any (resized to 224×224 during training)
- Recommended: 800×600 minimum for quality
- Minimum samples per class: 100 (recommended: 200+)

---

## 🏛️ UML Diagram

### Class Hierarchy & Module Dependencies

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CRICKET SHOT ANALYSIS SYSTEM                 │
└─────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      VIDEO PROCESSING LAYER                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────┐         ┌──────────────────┐              │
│  │  VideoProcessor     │         │  FrameExtractor  │              │
│  ├─────────────────────┤         ├──────────────────┤              │
│  │ - setup_capture()   │◄────────┤- extract_frames()              │
│  │ - setup_writer()    │         │- resize_frames()  │              │
│  │ - process_stream()  │         │- normalize()     │              │
│  └─────────────────────┘         └──────────────────┘              │
│           │                                                          │
│           └────────────────────────────────────────────┐            │
│                                                        ▼            │
│                                                   Output: Frames    │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                     POSE ESTIMATION LAYER                           │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    ┌──────────────────────────────────────┐                        │
│    │      PoseEstimator (MediaPipe)       │                        │
│    ├──────────────────────────────────────┤                        │
│    │ - mp_pose: mediapipe.Pose            │                        │
│    │ - min_detection_confidence: 0.5      │                        │
│    │ - min_tracking_confidence: 0.5       │                        │
│    │                                      │                        │
│    │ Methods:                             │                        │
│    │ - process_frame(frame) → results     │                        │
│    │ - get_landmarks(results) → 33 pts   │                        │
│    └──────────────────────────────────────┘                        │
│                     │                                               │
│                     ▼ (33 Landmarks: x, y, visibility)             │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION LAYER                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────┐                              │
│  │  BiomechanicalMetrics            │                              │
│  ├──────────────────────────────────┤                              │
│  │ - calculate_angle()              │                              │
│  │ - get_joint_coordinates()        │                              │
│  │ - calculate_metrics() ────────┐  │                              │
│  └──────────────────────────────┬┘  │                              │
│                                  │   │                              │
│  ┌────────────────────────────────┴──────────────────┐              │
│  │ Feature Vector (4 dimensions):                   │              │
│  │  ├─ elbow_angle: [0, 180]°                       │              │
│  │  ├─ spine_lean: [0, 180]°                        │              │
│  │  ├─ head_over_knee: [0, 1]                       │              │
│  │  └─ foot_direction: [0, 180]°                    │              │
│  └──────────────────────────────────────────────────┘              │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    SHOT CLASSIFICATION LAYER                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────┐   ┌──────────────────────────┐       │
│  │  ShotClassifier          │   │ ShotLandmarkClassifier   │       │
│  │  (ResNet18 - Image)      │   │ (Pose-based)             │       │
│  ├──────────────────────────┤   ├──────────────────────────┤       │
│  │ Input: Frame (224×224)   │   │ Input: 33 Landmarks     │       │
│  │ - model: ResNet18        │   │ - model: scikit-learn   │       │
│  │ - transform: ToTensor    │   │ - features: 8D vector   │       │
│  │                          │   │                          │       │
│  │ Output:                  │   │ Output:                  │       │
│  │ - shot_type: str         │   │ - shot_type: str        │       │
│  │ - confidence: float      │   │ - confidence: float     │       │
│  │ - probabilities: dict    │   │ - probabilities: dict   │       │
│  └──────────────────────────┘   └──────────────────────────┘       │
│                                                                      │
│  Classes: {Cover Drive, Straight Drive, Pull Shot, Cut Shot, ...}  │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                 SHOT QUALITY PREDICTION LAYER                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│    Feature Vector (4D) ──────┐                                      │
│                               ▼                                      │
│                 ┌───────────────────────────┐                        │
│                 │  StandardScaler           │                        │
│                 │  (Feature Normalization)  │                        │
│                 └───────────────────────────┘                        │
│                               │                                      │
│           ┌───────────────────┼───────────────────┬─────────────┐  │
│           │                   │                   │             │  │
│           ▼                   ▼                   ▼             ▼  │
│      ┌─────────┐         ┌──────────┐      ┌──────────┐   ┌─────────┐
│      │   SVM   │         │RandomFrst│      │Grad Boost│   │ XGBoost │
│      │Pipeline │         │Pipeline  │      │Pipeline  │   │Pipeline │
│      └────┬────┘         └────┬─────┘      └────┬─────┘   └────┬────┘
│           │                   │                   │             │
│           └───────────────────┼───────────────────┴─────────────┘
│                               │
│                    ┌──────────▼──────────┐
│                    │ Voting Ensemble     │
│                    │ (Average Probs)     │
│                    └──────────┬──────────┘
│                               │
│                               ▼
│                    Output: {label, confidence, probs}
│                    Classes: {Poor, Average, Good}
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                   RULE-BASED EVALUATION LAYER                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────┐                       │
│  │        Evaluation Scorer                 │                       │
│  ├──────────────────────────────────────────┤                       │
│  │ Input: All metrics per frame             │                       │
│  │                                          │                       │
│  │ Aspects Evaluated:                       │                       │
│  │  1. Footwork Score        [0-10]        │                       │
│  │  2. Head Position Score   [0-10]        │                       │
│  │  3. Swing Control Score   [0-10]        │                       │
│  │  4. Balance Score         [0-10]        │                       │
│  │  5. Follow-through Score  [0-10]        │                       │
│  │                                          │                       │
│  │ Output: evaluation.json                  │                       │
│  │ {                                        │                       │
│  │   "Footwork": {...},                     │                       │
│  │   "Head Position": {...},                │                       │
│  │   ...                                    │                       │
│  │ }                                        │                       │
│  └──────────────────────────────────────────┘                       │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                     VISUALIZATION LAYER                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────┐                │
│  │         OverlayUtils                            │                │
│  ├─────────────────────────────────────────────────┤                │
│  │ - draw_pose_landmarks(frame, results)           │                │
│  │ - display_metrics_on_frame(frame, metrics)      │                │
│  │                                                  │                │
│  │ Output: Frame with:                             │                │
│  │  • Skeleton overlay (joints + connections)      │                │
│  │  • Real-time metric text                        │                │
│  │  • Quality prediction label                     │                │
│  │  • Shot type classification                     │                │
│  └─────────────────────────────────────────────────┘                │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                        OUTPUT LAYER                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────┐  ┌──────────────────────┐                 │
│  │ Annotated Video      │  │ Result JSON Files   │                 │
│  ├──────────────────────┤  ├──────────────────────┤                 │
│  │ - {video}_ann.mp4    │  │ - {video}_result    │                 │
│  │ - 30 fps             │  │ - evaluation.json   │                 │
│  │ - Skeleton overlays  │  │ - ml_prediction.json│                 │
│  │ - Metric display     │  │                      │                 │
│  └──────────────────────┘  └──────────────────────┘                 │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                        API LAYER (FastAPI)                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────┐                        │
│  │         FastAPI Application            │                        │
│  ├────────────────────────────────────────┤                        │
│  │ GET  /health              ────────────┼────► Health check       │
│  │ POST /upload/             ────────────┼────► Process video      │
│  │ GET  /video/{video_stem}  ────────────┼────► Stream output      │
│  │ GET  /results/{video_stem}────────────┼────► Get results JSON   │
│  └────────────────────────────────────────┘                        │
│                                                                      │
│  Returns JSON with:                                                 │
│  {                                                                   │
│    "quality_prediction": {...},                                     │
│    "shot_type": "...",                                              │
│    "evaluation": {...},                                             │
│    "metrics_per_frame": [...]                                       │
│  }                                                                   │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### Step 1: Setup Environment

```bash
# Clone/navigate to project
cd cricket_cover_drive_analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Train Models (One-time Setup)

```bash
# Train shot quality prediction ensemble
python train_model.py

# Train shot type image classifier (optional)
python train_shot_model.py

# Train landmark-based shot classifier (optional)
python train_shot_landmark_model.py
```

**Output:** Trained models saved to `models/` directory

### Step 3: Analyze Videos

#### Option A: Command Line

```bash
# Analyze single video
python main.py test_folder/my_shot.mp4

# Batch process all videos in test_folder
# (main.py automatically processes all .mp4 files)
```

#### Option B: REST API

```bash
# Start API server
uvicorn backend.app:app --reload --port 8000

# Upload and analyze via curl
curl -X POST "http://localhost:8000/upload/" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@my_shot.mp4"

# Check results
curl "http://localhost:8000/results/my_shot"
```

#### Option C: Web Frontend

```bash
cd frontend
npm install
npm run dev

# Open browser to http://localhost:3000
```

### Step 4: View Results

```
output/
├── {video}_annotated.mp4        # Annotated video with overlays
├── {video}_result.json          # Complete analysis
├── evaluation.json              # Per-aspect scores
└── ml_prediction.json           # Quality prediction
```

---

## 📡 API Documentation

### REST Endpoints

#### 1. Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "ml_model": "ready",
  "shot_model": "ready"
}
```

---

#### 2. Upload & Analyze Video

```http
POST /upload/
Content-Type: multipart/form-data

Body: file=<binary video data>
```

**Request Example:**
```bash
curl -X POST "http://localhost:8000/upload/" \
  -F "file=@cricket_shot.mp4"
```

**Response (200 OK):**
```json
{
  "video_filename": "cricket_shot_annotated.mp4",
  "quality_prediction": {
    "label": "Good",
    "confidence": 0.967,
    "probabilities": {
      "Poor": 0.012,
      "Average": 0.021,
      "Good": 0.967
    }
  },
  "shot_type": "Cover Drive",
  "shot_type_confidence": 0.89,
  "evaluation": {
    "Footwork": {
      "score": 8.5,
      "feedback": "Excellent front foot positioning."
    },
    "Head Position": {
      "score": 9.0,
      "feedback": "Perfect head stability over front knee."
    },
    "Swing Control": {
      "score": 8.2,
      "feedback": "Good elbow height, well-controlled swing."
    },
    "Balance": {
      "score": 8.7,
      "feedback": "Excellent balance throughout the shot."
    },
    "Follow-through": {
      "score": 8.9,
      "feedback": "Complete follow-through with good extension."
    }
  },
  "metrics_summary": {
    "avg_elbow_angle": 135.4,
    "avg_spine_lean": 88.2,
    "avg_head_over_knee": 0.085,
    "avg_foot_direction": 92.1
  },
  "frame_count": 45,
  "processing_time_seconds": 12.3
}
```

---

#### 3. Stream Annotated Video

```http
GET /video/{video_stem}
```

**Example:**
```bash
curl "http://localhost:8000/video/cricket_shot" --output output.mp4
```

---

#### 4. Get Analysis Results

```http
GET /results/{video_stem}
```

**Example:**
```bash
curl "http://localhost:8000/results/cricket_shot"
```

**Response:** Same as upload response above

---

## 📈 Technical Metrics & Performance

### ML Model Performance

```
┌──────────────────────┬─────────────────┬──────────────┬──────────────┐
│ Model                │ CV Accuracy     │ CV Std Dev   │ Test Acc.    │
├──────────────────────┼─────────────────┼──────────────┼──────────────┤
│ SVM (RBF)            │ 94.3%           │ ±1.1%        │ 94.0%        │
│ Random Forest        │ 95.8%           │ ±0.9%        │ 95.3%        │
│ Gradient Boosting    │ 96.1%           │ ±0.8%        │ 95.7%        │
│ XGBoost              │ 96.6%           │ ±0.7%        │ 96.3%        │
│ Voting Ensemble ⭐   │ 97.1%           │ ±0.6%        │ 96.7%        │
└──────────────────────┴─────────────────┴──────────────┴──────────────┘
```

### Processing Speed

| Component | Time |
|-----------|------|
| Frame extraction (30 fps) | 0.5s |
| Pose estimation/frame | 15-20ms |
| Feature extraction/frame | 2-3ms |
| ML prediction | 1-2ms |
| Visualization | 5-10ms |
| **Total per 30-sec video** | ~12-15s |

### Memory Usage

- Pose model: ~30 MB
- ML ensemble: ~5 MB
- Shot classifier (ResNet18): ~45 MB
- **Total: ~80 MB RAM**

---

## 🔧 Configuration & Tuning

Edit `src/config.py` to adjust thresholds:

```python
# Example: Stricter head position scoring
HEAD_OVER_KNEE_GOOD_THRESHOLD = 0.05  # More strict

# Example: Adjust detection confidence
MIN_DETECTION_CONFIDENCE = 0.7  # Higher = more reliable but might miss

# Example: Change skill grading thresholds
ADVANCED_SCORE = 9  # Only very good shots grade as "Advanced"
```

---

## 📝 Summary of Technical Terms

| Term | Definition |
|------|-----------|
| **MediaPipe** | Google's ML framework for pose/hand/face detection |
| **Pose Landmark** | 33 detected body joint positions (x, y, visibility) |
| **Biomechanical Metric** | Computed feature (angle, distance) from pose landmarks |
| **Elbow Angle** | Angle formed by shoulder-elbow-wrist (degrees) |
| **Spine Lean** | Angle of spine from vertical (degrees) |
| **SVM** | Support Vector Machine - binary/multiclass classifier |
| **Random Forest** | Ensemble of decision trees with bootstrap aggregation |
| **Gradient Boosting** | Sequential tree ensemble with loss-based weighting |
| **XGBoost** | Extreme Gradient Boosting - optimized GB with regularization |
| **Voting Ensemble** | Meta-model combining multiple base models |
| **Soft Voting** | Average probability predictions from base models |
| **ResNet18** | 18-layer Residual Network for image classification |
| **Transfer Learning** | Fine-tuning pre-trained model on new task |
| **FastAPI** | Modern async Python web framework for APIs |
| **CORS** | Cross-Origin Resource Sharing for API security |
| **Confidence Score** | Probability [0,1] of predicted class |
| **Cross-Validation** | Technique for robust model evaluation on limited data |
| **Stratified KFold** | CV splitting maintaining class distribution |
| **Feature Normalization** | Scaling features to comparable ranges (e.g., StandardScaler) |

---

## 🐛 Troubleshooting

### MediaPipe Not Found
```bash
pip uninstall mediapipe -y
pip install mediapipe==0.10.11
```

### Models Not Found
```bash
python train_model.py                     # Train shot quality model
python train_shot_model.py                # Train shot type classifier
```

### Low Model Accuracy
- Check dataset quality and labeling
- Increase training epochs in `train_shot_model.py` (line ~55)
- Augment dataset with more real-world videos

### API Connection Issues
- Ensure CORS is enabled in `backend/app.py` (enabled by default)
- Check firewall allows port 8000
- Verify server is running: `curl http://localhost:8000/health`

---

## 📚 References & Resources

- **MediaPipe Pose:** https://mediapipe.dev/solutions/pose
- **scikit-learn Ensemble Methods:** https://scikit-learn.org/stable/modules/ensemble.html
- **XGBoost Documentation:** https://xgboost.readthedocs.io/
- **ResNet Paper:** He et al., "Deep Residual Learning for Image Recognition"
- **FastAPI Tutorial:** https://fastapi.tiangolo.com/

---

## 👥 Contributing

Contributions welcome! Please:
1. Create a feature branch
2. Test your changes thoroughly
3. Submit a pull request with description

---

## 📄 License

[Specify your license here]

---

**Last Updated:** April 2026  
**Version:** 2.0  
**Status:** Production Ready ✅

- Saves the best model to `models/best_model.pkl`
- Saves a comparison report to `output/model_comparison.json`

**Options:**

```bash
# Use your own labeled CSV
python train_model.py --dataset data/my_labeled_shots.csv

# Larger synthetic dataset
python train_model.py --samples 1200

# Save every trained model (not just the best)
python train_model.py --save-all
```

---

### 2 — Prepare a Real Labeled Dataset (optional but recommended)

The synthetic dataset is useful for a demo but real labeled data produces
better models. Here is the recommended workflow:

1. Place unlabeled videos in `test_folder/`
2. Run analysis with feature export enabled:
   ```bash
   python main.py --export-features
   ```
   This appends frame-level feature rows to `data/shot_quality_dataset.csv`
   with the `label` column empty.

3. Open the CSV in any spreadsheet editor and fill in the `label` column:
   - `Poor` — clear technical errors (collapsed arm, wrong foot direction, etc.)
   - `Average` — partially correct technique with some flaws
   - `Good` — textbook cover drive with all metrics in range

4. Re-train the model on the real data:
   ```bash
   python train_model.py --dataset data/shot_quality_dataset.csv
   ```

---

### 3 — Run Video Analysis

```bash
# Analyze all videos in test_folder/
python main.py

# Analyze a single file
python main.py --video path/to/cover_drive.mp4
```

The pipeline:
1. Extracts frames and detects 33 body landmarks (MediaPipe)
2. Computes four biomechanical features per frame:
   - `elbow_angle` — swing control
   - `spine_lean` — balance
   - `head_over_knee` — head position
   - `foot_direction` — footwork
3. Averages features across all detected frames
4. Runs the trained ML model on the aggregated feature vector
5. Writes results to `output/`

**Output files:**

| File                          | Description                                      |
|-------------------------------|--------------------------------------------------|
| `output/<name>_annotated.mp4` | Video with skeleton overlay and metric text      |
| `output/evaluation.json`      | Rule-based per-aspect scores (0–10)              |
| `output/ml_prediction.json`   | ML label, class probabilities, and confidence    |

**Example `ml_prediction.json`:**

```json
{
    "label": "Good",
    "probabilities": {
        "Poor": 0.03,
        "Average": 0.09,
        "Good": 0.88
    },
    "confidence": 0.88,
    "model_used": "Ensemble",
    "frame_count": 142,
    "mean_features": {
        "elbow_angle": 127.4,
        "spine_lean": 91.2,
        "head_over_knee": 0.043,
        "foot_direction": 89.7
    }
}
```

---

### 4 — Run the FastAPI Backend

```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Available endpoints:**

| Method | Path                    | Description                               |
|--------|-------------------------|-------------------------------------------|
| GET    | `/health`               | Server + model status                     |
| GET    | `/model/info`           | Loaded model name and feature list        |
| GET    | `/model/comparison`     | Full model comparison JSON                |
| POST   | `/upload/`              | Upload video → annotated video + ML result |
| GET    | `/video/{filename}`     | Stream an annotated video                 |
| GET    | `/results/{video_stem}` | Fetch saved prediction results            |

**Example upload:**

```bash
curl -X POST http://127.0.0.1:8000/upload/ \
     -F "file=@cover_drive.mp4"
```

Response:
```json
{
  "message": "Processed",
  "filename": "cover_drive_annotated.mp4",
  "ml_prediction": {
    "label": "Good",
    "confidence": 0.88,
    "probabilities": {"Poor": 0.03, "Average": 0.09, "Good": 0.88},
    "model_used": "Ensemble"
  },
  "evaluation": {
    "Footwork":       {"score": 8.5, "feedback": "..."},
    "Head Position":  {"score": 7.2, "feedback": "..."},
    "Swing Control":  {"score": 9.0, "feedback": "..."},
    "Balance":        {"score": 8.1, "feedback": "..."},
    "Follow-through": {"score": 7.6, "feedback": "..."}
  }
}
```

---

## ML Model Details

### Features

All four input features are derived from MediaPipe pose landmarks:

| Feature           | Description                                     | Unit    |
|-------------------|-------------------------------------------------|---------|
| `elbow_angle`     | Angle at the left elbow joint                   | degrees |
| `spine_lean`      | Angle between shoulder, hip, and vertical axis  | degrees |
| `head_over_knee`  | Normalised horizontal distance nose → knee      | 0–1     |
| `foot_direction`  | Angle of the front foot relative to ankle       | degrees |

### Models

| Model              | Type                | Key hyperparameters                           |
|--------------------|---------------------|-----------------------------------------------|
| SVM                | Kernel SVM          | RBF kernel, C=10, gamma=scale                 |
| RandomForest       | Bagging ensemble    | 200 trees, no max depth                       |
| GradientBoosting   | Boosting ensemble   | 200 estimators, lr=0.05, max_depth=4          |
| XGBoost            | Boosting ensemble   | 200 rounds, lr=0.05, max_depth=4              |
| Ensemble           | Soft voting         | Combines all four models                      |

All models are wrapped in scikit-learn `Pipeline` objects that include a
`StandardScaler` step, ensuring features are normalised before reaching any
classifier.

### Aggregation Strategy

Frame-level predictions are noisy due to landmark detection jitter.
Instead of voting across frames, the pipeline averages feature values across
all valid frames and predicts once on the mean feature vector. This produces
a stable, video-level quality label.

---

## Supported Video Sources

- Local files: `.mp4`, `.avi`, `.mov`
- Remote URLs supported by `yt-dlp` (via `src/video_processing.py`)

---

## Notes

- Run `python train_model.py` before `python main.py` or starting the backend.
- The `models/` directory is excluded from version control (add to `.gitignore`).
- `soft_c.py` is deprecated and can be deleted; it is no longer imported anywhere.
- The repo does not currently include a Streamlit frontend.