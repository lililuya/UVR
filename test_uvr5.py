import soundfile as sf
import torch
import sys
from vr import AudioPre
import os
import ffmpeg
import traceback

weight_uvr5_root = "./uvr5_weights"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or name.endswith(".ckpt") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", "").replace(".ckpt", ""))

def clean_path(path_str:str):
    if path_str.endswith(('\\','/')):
        return clean_path(path_str[0:-1])
    path_str = path_str.replace('/', os.sep).replace('\\', os.sep)
    return path_str.strip(" ").strip('\'').strip("\n").strip('"').strip(" ").strip("\u202a")


""" invoke the UVR5 model to split vocals and instruments
    Parameters:
        model_name(str)      : the name of the model to be used, refer to https://huggingface.co/seanghay/uvr_models/tree/main, default is "1_HP-UVR.pth"
        inp_root(str)        : the directory of input audio files
        wav_inputs(list[str]): the path list of input audio files
        
        save_root_vocal(str) : the directory to save the vocal audio files
        save_root_ins(str)   : the directory to save the instrument audio files
        agg(int)             : degree of aggressiveness in vocal extraction, higher value means more clean vocal. default is 10
        format(str)          : the format of the output audio files, default is "flac", support ["wav", "flac", "mp3", "m4a"]
"""
def uvr(model_name: str, inp_root: str, wav_inputs: list[str], save_root_vocal: str, save_root_ins:str, agg: int, format: str, device = "cuda:0", is_half="True"):
    infos = []
    inp_root = clean_path(inp_root)
    save_root_vocal = clean_path(save_root_vocal)
    save_root_ins = clean_path(save_root_ins)
    is_hp3 = "HP3" in model_name
    print(f"choose the model {model_name}")
    func = AudioPre
    pre_fun = func(
        agg=int(agg),
        model_path=os.path.join(weight_uvr5_root, model_name + ".pth"),
        device=device,
        is_half=is_half,
    )
    if inp_root != "": 
        paths = [os.path.join(inp_root, name) for name in os.listdir(inp_root)]
    else:
        paths = [path for path in wav_inputs]
    for path in paths:
        inp_path = os.path.join(inp_root, path)
        if(os.path.isfile(inp_path)==False):
            continue
        need_reformat = 1
        done = 0
        try:
            info = ffmpeg.probe(inp_path, cmd="ffprobe")
            if (
                info["streams"][0]["channels"] == 2
                and info["streams"][0]["sample_rate"] == "44100"
            ):
                need_reformat = 0 
                pre_fun._path_audio_(
                    inp_path, save_root_ins, save_root_vocal, format, is_hp3
                )
                done = 1
        except:
            need_reformat = 1
            traceback.print_exc()
        if need_reformat == 1:
            current_directory = os.getcwd()
            temp_dir = os.path.join(current_directory, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            tmp_path = "%s/%s.reformatted.wav" % (
                temp_dir,
                os.path.basename(inp_path),
            )
            os.system(
                f'ffmpeg -i "{inp_path}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
            )
            inp_path = tmp_path
        try:
            if done == 0:
                pre_fun._path_audio_(
                    inp_path, save_root_ins, save_root_vocal, format, is_hp3
                )
            infos.append("%s->Success" % (os.path.basename(inp_path)))
        except:
            infos.append(
                "%s->%s" % (os.path.basename(inp_path), traceback.format_exc())
            )
            print(infos)
    infos.append(traceback.format_exc())
    try:
        if model_name == "onnx_dereverb_By_FoxJoy":
            del pre_fun.pred.model
            del pre_fun.pred.model_
        else:
            del pre_fun.model
            del pre_fun
    except:
        traceback.print_exc()
    print("clean_empty_cache")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
        
if __name__=="__main__":
    format = "wav"
    model_choose = "1_HP-UVR"
    opt_vocal_root = "/mnt/hd1/Weather/GPT-SoVITS/UVR_project/test_data/processed/instrument"
    opt_ins_root   = "/mnt/hd1/Weather/GPT-SoVITS/UVR_project/test_data/processed/vocal"
    os.makedirs(opt_vocal_root, exist_ok=True)
    os.makedirs(opt_ins_root, exist_ok=True)
    wav_inputs = ""
    dir_wav_input = "/mnt/hd1/Weather/GPT-SoVITS/UVR_project/test_data/"
    agg = "10"
    uvr(model_choose, dir_wav_input, wav_inputs, opt_vocal_root, opt_ins_root, agg, format)
