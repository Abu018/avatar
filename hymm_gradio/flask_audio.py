import os
import numpy as np
import torch
import warnings
import threading
import traceback
import uuid
import uvicorn
from fastapi import FastAPI, Body
from pathlib import Path
from datetime import datetime
import torch.distributed as dist
from hymm_gradio.tool_for_end2end import *
from hymm_sp.config import parse_args
from hymm_sp.sample_inference_audio import HunyuanVideoSampler

from hymm_sp.modules.parallel_states import (
    initialize_distributed,
    nccl_info,
)

from transformers import WhisperModel
from transformers import AutoFeatureExtractor
from hymm_sp.data_kits.face_align import AlignImage
from hymm_gradio.tts_service import synthesize_to_wav
from hymm_gradio.wav2lip_service import run_wav2lip


warnings.filterwarnings("ignore")
MODEL_OUTPUT_PATH = os.environ.get('MODEL_BASE')
AVATAR_STORE_DIR = os.environ.get("AVATAR_STORE_DIR", "./assets/avatars")
app = FastAPI()
rlock = threading.RLock()


def _ensure_avatar_store():
    os.makedirs(AVATAR_STORE_DIR, exist_ok=True)


def _avatar_path(avatar_id):
    return os.path.join(AVATAR_STORE_DIR, f"{avatar_id}.mp4")


def save_avatar_video_from_base64(video_base64, avatar_id=None):
    _ensure_avatar_store()
    final_avatar_id = avatar_id or str(uuid.uuid4())
    output_path = _avatar_path(final_avatar_id)
    save_video_base64_to_local(video_path=None, base64_buffer=video_base64, output_video_path=output_path)
    return final_avatar_id, output_path


def get_avatar_video_path(avatar_id):
    if not avatar_id:
        return None
    path = _avatar_path(avatar_id)
    return path if os.path.exists(path) else None


@app.get('/avatars/list')
def list_avatars():
    try:
        _ensure_avatar_store()
        avatars = []
        for name in sorted(os.listdir(AVATAR_STORE_DIR)):
            if not name.endswith(".mp4"):
                continue
            avatar_id = name[:-4]
            avatar_path = os.path.join(AVATAR_STORE_DIR, name)
            avatars.append({"avatar_id": avatar_id, "avatar_path": avatar_path})
        return {"errCode": 0, "avatars": avatars, "info": "succeed"}
    except Exception:
        traceback.print_exc()
        return {"errCode": -1, "avatars": [], "info": "failed to list avatars"}


def run_tts(text, voice_id=None, language=None, speed=1.0, provider=None):
    return synthesize_to_wav(
        text=text,
        voice_id=voice_id or "aura-2-thalia-en",
        language=language,
        speed=speed,
        provider=provider or "deepgram",
    )


@app.post('/tts')
def tts(data=Body(...)):
    """
    TTS endpoint scaffold.
    Expected payload:
    {
      "text_input": "hello world",
      "voice_id": "optional",
      "language": "optional",
      "speed": 1.0,
      "provider": "optional"
    }
    """
    try:
        text = data.get("text_input", None)
        if text is None:
            text = data.get("text", None)
        if not text:
            return {"errCode": -3, "audio_path": None, "info": "text_input is required"}

        voice_id = data.get("voice_id", None)
        language = data.get("language", None)
        speed = data.get("speed", 1.0)
        provider = data.get("provider", None)

        audio_path = run_tts(
            text=text,
            voice_id=voice_id,
            language=language,
            speed=speed,
            provider=provider,
        )
        return {"errCode": 0, "audio_path": audio_path, "info": "succeed"}
    except NotImplementedError as e:
        return {"errCode": -4, "audio_path": None, "info": str(e)}
    except Exception:
        traceback.print_exc()
        return {"errCode": -1, "audio_path": None, "info": "failed to generate audio"}


@app.api_route('/predict2', methods=['GET', 'POST'])
def predict(data=Body(...)):
    is_acquire = False
    error_info = ""
    try:
        is_acquire = rlock.acquire(blocking=False)
        if is_acquire:
            res = predict_wrap(data)
            return res
    except Exception as e:
        error_info = traceback.format_exc()
        print(error_info)
    finally:
        if is_acquire:
            rlock.release()
    return {"errCode": -1, "info": "broken"}

@app.post('/wav2lip')
def wav2lip_api(data=Body(...)):
    try:
        decoded = process_input_dict(data)
        base_video_path = decoded.get("base_video_path", None)
        audio_path = decoded.get("audio_path", None)
        avatar_id = data.get("avatar_id", None)
        if base_video_path is None and avatar_id is not None:
            base_video_path = get_avatar_video_path(avatar_id)

        text_input = data.get("text_input", None) or data.get("tts_text", None)
        if audio_path is None and text_input:
            audio_path = run_tts(
                text=text_input,
                voice_id=data.get("voice_id", None),
                language=data.get("language", None),
                speed=data.get("speed", 1.0),
                provider=data.get("provider", "deepgram"),
            )

        if base_video_path is None or audio_path is None:
            return {
                "errCode": -3,
                "content": [{"buffer": None}],
                "info": "base_video_buffer/video_buffer and audio_buffer or text_input are required",
            }

        output_path = run_wav2lip(
            face_video_path=base_video_path,
            audio_path=audio_path,
            checkpoint_path=data.get("wav2lip_checkpoint", None),
            wav2lip_repo=data.get("wav2lip_repo", None),
        )
        video_b64 = encode_video_to_base64(output_path)
        return {
            "errCode": 0,
            "content": [{"buffer": video_b64}],
            "info": "succeed",
            "avatar_id": avatar_id,
            "avatar_path": base_video_path,
        }
    except Exception:
        traceback.print_exc()
        return {"errCode": -1, "content": [{"buffer": None}], "info": "failed to run wav2lip"}


def predict_wrap(input_dict={}):
    rank = local_rank = 0
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        rank = local_rank = torch.distributed.get_rank()
        print(f"sp_size={nccl_info.sp_size}, rank {rank} local_rank {local_rank}")
    try:
        print(f"----- rank = {rank}")
        if rank == 0:
            raw_input = dict(input_dict)
            input_dict = process_input_dict(input_dict)

            print('------- start to predict -------')
            # Parse input arguments
            image_path = input_dict["image_path"]
            driving_audio_path = input_dict["audio_path"]
            text_input = raw_input.get("text_input", None) or raw_input.get("tts_text", None)

            prompt = input_dict["prompt"]

            save_fps = input_dict.get("save_fps", 25)
            tts_generated_audio_path = None


            ret_dict = None
            if driving_audio_path is None and text_input:
                try:
                    tts_generated_audio_path = run_tts(
                        text=text_input,
                        voice_id=raw_input.get("voice_id", None),
                        language=raw_input.get("language", None),
                        speed=raw_input.get("speed", 1.0),
                        provider=raw_input.get("provider", "deepgram"),
                    )
                    driving_audio_path = tts_generated_audio_path
                except Exception as e:
                    print(f"TTS generation failed: {e}")
                    return {
                        "errCode": -5,
                        "content": [{"buffer": None}],
                        "info": f"failed to generate tts audio: {e}",
                    }

            if image_path is None or driving_audio_path is None:
                ret_dict = {
                    "errCode": -3, 
                    "content": [
                        {
                            "buffer": None
                        },
                    ], 
                    "info": "input content is not valid", 
                }

                print(f"errCode: -3, input content is not valid!")
                return ret_dict

            # Preprocess input batch
            torch.cuda.synchronize()

            a = datetime.now()
            
            try:
                model_kwargs_tmp = data_preprocess_server(
                                        args, image_path, driving_audio_path, prompt, feature_extractor
                                        )
            except:
                ret_dict = {
                    "errCode": -2,         
                    "content": [
                            {
                                "buffer": None
                            },
                        ],
                    "info": "failed to preprocess input data"
                }
                print(f"errCode: -2, preprocess failed!")
                return ret_dict

            text_prompt = model_kwargs_tmp["text_prompt"]
            audio_path = model_kwargs_tmp["audio_path"]
            image_path = model_kwargs_tmp["image_path"]
            fps = model_kwargs_tmp["fps"]
            audio_prompts = model_kwargs_tmp["audio_prompts"]
            audio_len = model_kwargs_tmp["audio_len"]
            motion_bucket_id_exps = model_kwargs_tmp["motion_bucket_id_exps"]
            motion_bucket_id_heads = model_kwargs_tmp["motion_bucket_id_heads"]
            pixel_value_ref = model_kwargs_tmp["pixel_value_ref"]
            pixel_value_ref_llava = model_kwargs_tmp["pixel_value_ref_llava"]
            


            torch.cuda.synchronize()
            b = datetime.now()
            preprocess_time = (b - a).total_seconds()
            print("="*100)
            print("preprocess time :", preprocess_time)
            print("="*100)
            
        else:
            text_prompt = None
            audio_path = None
            image_path = None
            fps = None
            audio_prompts = None
            audio_len = None
            motion_bucket_id_exps = None
            motion_bucket_id_heads = None
            pixel_value_ref = None
            pixel_value_ref_llava = None

    except:
        traceback.print_exc()
        if rank == 0:
            ret_dict = {
                "errCode": -1,         # Failed to generate video
                "content":[
                    {
                        "buffer": None
                    }
                ],
                "info": "failed to preprocess",
            }
            return ret_dict

    try:
        broadcast_params = [
            text_prompt,
            audio_path,
            image_path,
            fps,
            audio_prompts,
            audio_len,
            motion_bucket_id_exps,
            motion_bucket_id_heads,
            pixel_value_ref,
            pixel_value_ref_llava,
        ]
        dist.broadcast_object_list(broadcast_params, src=0)
        outputs = generate_image_parallel(*broadcast_params)

        if rank == 0:
            samples = outputs["samples"]
            sample = samples[0].unsqueeze(0)

            sample = sample[:, :, :audio_len[0]]
            
            video = sample[0].permute(1, 2, 3, 0).clamp(0, 1).numpy()
            video = (video * 255.).astype(np.uint8)

            output_dict = {
                "err_code": 0, 
                "err_msg": "succeed", 
                "video": video, 
                "audio": tts_generated_audio_path or input_dict.get("audio_path", None),
                "save_fps": save_fps, 
            }

            ret_dict = process_output_dict(output_dict)
            try:
                avatar_id = raw_input.get("avatar_id", None)
                avatar_video_b64 = ret_dict["content"][0]["buffer"]
                saved_avatar_id, saved_avatar_path = save_avatar_video_from_base64(
                    avatar_video_b64,
                    avatar_id=avatar_id,
                )
                ret_dict["avatar_id"] = saved_avatar_id
                ret_dict["avatar_path"] = saved_avatar_path
            except Exception as e:
                print(f"Warning: failed to persist avatar video: {e}")
            return ret_dict
    
    except:
        traceback.print_exc()
        if rank == 0:
            ret_dict = {
                "errCode": -1,         # Failed to generate video
                "content":[
                    {
                        "buffer": None
                    }
                ],
                "info": "failed to generate video",
            }
            return ret_dict
        
    return None
    
def generate_image_parallel(text_prompt,
                    audio_path,
                    image_path,
                    fps,
                    audio_prompts,
                    audio_len,
                    motion_bucket_id_exps,
                    motion_bucket_id_heads,
                    pixel_value_ref,
                    pixel_value_ref_llava
                    ):
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")

    batch = {
        "text_prompt": text_prompt,
        "audio_path": audio_path,
        "image_path": image_path,
        "fps": fps,
        "audio_prompts": audio_prompts,
        "audio_len": audio_len,
        "motion_bucket_id_exps": motion_bucket_id_exps,
        "motion_bucket_id_heads": motion_bucket_id_heads,
        "pixel_value_ref": pixel_value_ref,
        "pixel_value_ref_llava": pixel_value_ref_llava
    }

    samples = hunyuan_sampler.predict(args, batch, wav2vec, feature_extractor, align_instance)
    return samples

def worker_loop():
    while True:
        predict_wrap()
        

if __name__ == "__main__":
    audio_args = parse_args()
    initialize_distributed(audio_args.seed)
    hunyuan_sampler = HunyuanVideoSampler.from_pretrained(
        audio_args.ckpt, args=audio_args)
    args = hunyuan_sampler.args
    
    rank = local_rank = 0
    device = torch.device("cuda")
    if nccl_info.sp_size > 1:
        device = torch.device(f"cuda:{torch.distributed.get_rank()}")
        rank = local_rank = torch.distributed.get_rank()

    feature_extractor = AutoFeatureExtractor.from_pretrained(f"{MODEL_OUTPUT_PATH}/ckpts/whisper-tiny/")
    wav2vec = WhisperModel.from_pretrained(f"{MODEL_OUTPUT_PATH}/ckpts/whisper-tiny/").to(device=device, dtype=torch.float32)
    wav2vec.requires_grad_(False)


    BASE_DIR = f'{MODEL_OUTPUT_PATH}/ckpts/det_align/'
    det_path = os.path.join(BASE_DIR, 'detface.pt')    
    align_instance = AlignImage("cuda", det_path=det_path)



    if rank == 0:
        uvicorn.run(app, host="0.0.0.0", port=80)
    else:
        worker_loop()
    
