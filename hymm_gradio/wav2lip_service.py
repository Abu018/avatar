import os
import uuid
import subprocess


TEMP_DIR = "./temp"
DEFAULT_PYTHON_BIN = os.environ.get("WAV2LIP_PYTHON_BIN", "python")
DEFAULT_WAV2LIP_REPO = os.environ.get("WAV2LIP_REPO", "")
DEFAULT_WAV2LIP_SCRIPT = os.environ.get("WAV2LIP_INFER_SCRIPT", "inference.py")
DEFAULT_WAV2LIP_CHECKPOINT = os.environ.get("WAV2LIP_CHECKPOINT", "")


def run_wav2lip(
    face_video_path: str,
    audio_path: str,
    output_path: str = None,
    checkpoint_path: str = None,
    wav2lip_repo: str = None,
):
    if not face_video_path or not os.path.exists(face_video_path):
        raise ValueError("face_video_path is required and must exist")
    if not audio_path or not os.path.exists(audio_path):
        raise ValueError("audio_path is required and must exist")

    repo_dir = wav2lip_repo or DEFAULT_WAV2LIP_REPO
    if not repo_dir:
        raise ValueError("WAV2LIP_REPO is not set")
    if not os.path.isdir(repo_dir):
        raise ValueError(f"WAV2LIP_REPO does not exist: {repo_dir}")

    ckpt = checkpoint_path or DEFAULT_WAV2LIP_CHECKPOINT
    if not ckpt:
        raise ValueError("WAV2LIP_CHECKPOINT is not set")
    if not os.path.exists(ckpt):
        raise ValueError(f"WAV2LIP_CHECKPOINT does not exist: {ckpt}")

    os.makedirs(TEMP_DIR, exist_ok=True)
    if output_path is None:
        output_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_wav2lip.mp4")

    infer_script = os.path.join(repo_dir, DEFAULT_WAV2LIP_SCRIPT)
    if not os.path.exists(infer_script):
        raise ValueError(f"Wav2Lip inference script not found: {infer_script}")

    cmd = [
        DEFAULT_PYTHON_BIN,
        infer_script,
        "--checkpoint_path",
        ckpt,
        "--face",
        face_video_path,
        "--audio",
        audio_path,
        "--outfile",
        output_path,
    ]
    subprocess.run(cmd, cwd=repo_dir, check=True)
    return output_path
