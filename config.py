from pathlib import Path

class ProjectPaths:
    # Root path
    ROOT = Path(__file__).parent
    
    # Main directories
    ENVIRONMENTS = ROOT / "environments"
    PARAMETERS = ROOT / "parameters"
    EVALUATION = ROOT / "evaluation"
    TRAINING = ROOT / "training"
    TEACHER_STUDENT = ROOT / "teacher_student"
    
    # Environment files
    CUSTOM_ENV_XML = ENVIRONMENTS / "custom_env.xml"
    CUSTOM_ENV_DEBUG_XML = ENVIRONMENTS / "custom_env_debug_wall.xml"
    STAIRS_ENV_XML = ENVIRONMENTS / "stairs_env.xml"
    
    # Parameter files
    PARAMS = PARAMETERS / "params.npy"
    PARAMS_BASELINE = PARAMETERS / "params_baseline.npy"
    PARAMS_WITH_HEIGHT = PARAMETERS / "params_with_height.npy"
    PARAMS_WITH_HEIGHT_AND_KNEE = PARAMETERS / "params_with_height_and_knee.npy"
    PARAMS_WITH_KNEE = PARAMETERS / "params_with_knee.npy"
    TEACHER_PARAMS = PARAMETERS / "teacher_params.npy"