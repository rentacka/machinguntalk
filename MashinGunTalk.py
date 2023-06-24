#!/usr/bin/env python
# coding: utf-8

# In[2]:


import gradio as gr
import numpy as np
import os
import json
import requests

import torch
import torchaudio
from transformers import AutoTokenizer, AutoModelForCausalLM

from faster_whisper import WhisperModel
from moviepy.editor import *
#from moviepy.editor import VideoFileClip
from moviepy.video.tools.subtitles import *
from datetime import timedelta
from srt import Subtitle
import srt

from pathlib import Path

from VGwebui.tts.espnet_interface import load_model,generate_speech
from VGwebui.vc.vc_interface import load_audio,load_wav,convert_voice

from scipy.io.wavfile import write
from pydub import AudioSegment, playback,audio_segment


# In[3]:


#アルゴリズムのインスタンス生成
#rinna_AI = "models/japanese-gpt-1b"
rinna_AI = "rinna/japanese-gpt-1b"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer_rinna = AutoTokenizer.from_pretrained(rinna_AI, use_fast=False)
model_rinna = AutoModelForCausalLM.from_pretrained(rinna_AI).to(device)

# or run on GPU with INT8
model = WhisperModel("models/whisper-large-v2-ct2", device="cuda", compute_type="int8_float16")

#"男性": "finetune50epoch_jvs001_jsut","女性": "finetune50epoch_jvs010_jsut"
# lang:"en": "vctk","ja": "jvs"
net_g = load_model("finetune50epoch_jvs010_jsut")
#net_g = load_model("jvs")


# In[4]:


def generate_speech(model, text, speed):
    model.decode_conf.update({'alpha': 1 / speed})
    with torch.no_grad():
        wav = model(text)

    return (22050, wav["wav"].view(-1).cpu().numpy())

def coeiro_old_wav(text,speed):    
    samplerate,np_wav=generate_speech(net_g,text,speed # 'alpha': 1 / speed
                   )
    t = np.linspace(0., 1., samplerate)
    amplitude = np.iinfo(np.int16).max
    data = amplitude * np_wav/np.max(np.abs(np_wav))
    
    tmp_path = "tmp_wav/"+text+".wav"
    write(tmp_path, samplerate, data.astype(np.int16))    
    
    return tmp_path

def subs_to_wav(subs):
    # 全音声時間を足した無音ファイルを作成
    start_time = subs[0].start.total_seconds()
    end_time = subs[-1].end.total_seconds()
    duration = (end_time - start_time)*1000
    silent_segment = AudioSegment.silent(duration=duration)
        
    for sub in subs:
        if(len(sub.content) > 0):
            wav = coeiro_old_wav(sub.content,1)
            seg = AudioSegment.from_wav(wav)
            pos = sub.start.total_seconds()*1000
            silent_segment = silent_segment.overlay(seg, position=pos)
            return silent_segment    
    return ""

def CompositeMp4(mp4file,mixed_sound,srt_file,outputMP4):
    # 動画ファイルを読み込む
    video = VideoFileClip(mp4file)
    # Waveファイルを読み込む
    audio = AudioFileClip(mixed_sound)
    # 字幕ファイルを読み込む
    subtitles = SubtitlesClip(srt_file)
    #元の動画の音声を読み込み
    orig_audio = video.audio
    
    # mixed_sounds.wavファイルを動画に埋め込む
    video = video.set_audio(audio)
    #元の動画の音声を埋め込み
    video = video.set_audio(audio)
    # 字幕を動画に埋め込む
    video = CompositeVideoClip([video, subtitles.set_position(('center', 'bottom'))])

    # 出力ファイルとして保存する
    video.write_videofile(outputMP4, codec="libx264")
    
def rinnaText(prompt,max=5000):
    # りんなでテキスト生成
    input_ids2 = tokenizer_rinna.encode(prompt,  add_special_tokens=False, return_tensors="pt")
    with torch.no_grad():
        output_ids2 = model_rinna.generate(input_ids2.to(model_rinna.device), do_sample=True,
                                          max_length=max,
            pad_token_id=tokenizer_rinna.pad_token_id,
            bos_token_id=tokenizer_rinna.bos_token_id,
            eos_token_id=tokenizer_rinna.eos_token_id,
            bad_words_ids=[[tokenizer_rinna.unk_token_id]])
    output_text = tokenizer_rinna.decode(output_ids2[0])

#    generated_text = output_text.replace(prompt, '')
    delchars = len(prompt) # 削除する文字数
    generated_text = output_text[delchars:] # 削除結果を元の変数に代入

    return generated_text.rstrip("</s>")

def get_speed(subtitle):
    # 字幕の開始時刻と終了時刻を取得する
    start_time = subtitle.start
    end_time = subtitle.end

    # 字幕の長さを取得する
    length = len(subtitle.text)

    # 字幕の速度を求める
    speed = length / (end_time - start_time)

    return speed

def add_line(s):
    new_s = s
    s_count = len(s)
    s_max_count = 15
    if s_count >= s_max_count:
        if (s_count - s_max_count) >= 3:
            # 15文字以上、かつ、2行目が3文字以上あれば、改行する
            # つまり、18文字以上であれば、15文字で改行する
            new_s = s[:s_max_count] + "\n" + s[s_max_count:]
    return new_s

def write_subtitles(segments,file_name):
    subs = []
    for data in segments:
        index = data.id + 1
        start = data.start
        end = data.end
        text = add_line(data.text)
        #text = data.text
        sub = Subtitle(index=index, start=timedelta(seconds=timedelta(seconds=start).seconds,
                                            microseconds=timedelta(seconds=start).microseconds),
                   end=timedelta(seconds=timedelta(seconds=end).seconds,
                                 microseconds=timedelta(seconds=end).microseconds), content=text, proprietary='')
        subs.append(sub)
        #print(text)
    
    with open(file_name, mode="w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
        


# In[10]:


#UI
class MyInit:
    def __init__(self):
        self.end = 0

def inferenceSSS(filepath,is_selif,is_origZimakuChechbox):
    subtitles, info = model.transcribe(filepath, beam_size=5)
    new_Addsubtitles = []
    if(is_selif==True):
        befo_sub=MyInit()
        befo_sub.end=0
        # 字幕ごとに新しいセリフを生成する
        for subtitle in subtitles:
            # 平均トークン速度を求める
            zimakuSpeed = get_speed(subtitle)
            #前回の字幕と今回の字幕の間のトークン時間
            time_between_subtitles = befo_sub.end - subtitle.start
            #空白時間が1秒より大きいなら
            if time_between_subtitles >= 1:
                print("[%.2fs -> %.2fs] %s" % (befo_sub.end,subtitle.start, "セリフ生成中"))
                #りんなモデルで字幕生成
                new_subtitle = rinnaText(subtitle,zimakuSpeed*time_between_subtitles)
            
                # 新しい字幕をリストに追加する
                new_Addsubtitles.append(new_subtitle)
            if is_origZimakuChechbox == True:
                #元の字幕を追加
                new_Addsubtitles.append(subtitle)
                
            befo_sub=subtitle

        # 字幕を保存
        write_subtitles(new_Addsubtitles, "new_addsubtitles.srt")
    else:
        #字幕を保存
        print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

        for segment in subtitles:
            print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        write_subtitles(subtitles, "subtitles.srt")

    if(filepath.endswith(".mp4") and is_selif==True):
        # 字幕から音声作成
        wav = subs_to_wav(new_Addsubtitles)
        
        #mp4に字幕と音声を一つの動画にする(mp4file,mixed_sound,srt_file,outputMP4)
        CompositeMp4(filepath,wav,"new_addsubtitles.srt","output.mp4")
        
with gr.Blocks() as demo:
        input = gr.Textbox(label="メディアファイル(mp4,mp3,wav")
        selifChechbox = gr.Checkbox(label='マシンガントーク生成', key='-Selif-',value = True)
        origZimakuChechbox = gr.Checkbox(label='元々の動画の字幕も生成(mp4限定', key='-origZimaku-',value = False)
        greet_btn = gr.Button("字幕付きメディア生成")
        
        output = gr.Textbox(label="出力")
        greet_btn.click(fn=inferenceSSS,inputs=[input,selifChechbox,origZimakuChechbox],outputs=output)


demo.launch()


# In[4]:


#Test Mp4から字幕を作成
video_path="「FateGrand Order」第2部後期オープニングムービー.mp4"
video = VideoFileClip(video_path)
video.audio.write_audiofile("temp_audio.mp3")

segments, info = model.transcribe("temp_audio.mp3", beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))


# In[ ]:


#tts_interface使用時
#非推奨
def coeiro_wav(text,speed):
    clean_text,tts_audio=generate_speech(net_g,
                    "jvs",# lang:"en": "vctk","ja": "jvs"
                    text,
                    1,# sid:speaker_list.index tts/models/{lang_dic[lang]}_speakers.txt
                    False,#True,False Textの履歴クリア
                    speed#Scale minimum=0.1, maximum=2
                   )
    return tts_audio

#非推奨
def coeiro_tts2rvc_wav(text):
    _, tts_audio = generate_speech(net_g,
                    "jvs",# lang:"en": "vctk","ja": "jvs"
                    text,
                    0,# sid:speaker_list.index tts/models/{lang_dic[lang]}_speakers.txt
                    False,#True,False Textの履歴クリア
                    0.1#Scale minimum=0.1, maximum=2
                   )
    if vcid != 'No conversion':# cleaned=vcid:'No conversion' vc/models/名前
        return vc_interface.convert_voice(hubert_model, vc, net_g, tts_audio, vcid, pitch, f0method)

    return tts_audio

#非推奨
def read_subtitles(file_path):
    with open(file_path, 'r') as file:
        subtitles = file.read()
    return subtitles

#非推奨
def subtitle_to_subs(subtitle_file):
    subtitle_text=""
    
    subs2=[]
    # 字幕ファイルの読み込み
    with open(subtitle_file, mode='r', encoding="utf-8") as f: 
        subs = srt.parse(f.read()) 
        for sub in subs:
            if(len(sub.content) > 0):
                 subs2.append(sub)

    return subs2

#非推奨
def subs_to_AudioSegments(subs):
    if not os.path.isdir("tmp_wav"):
        os.makedirs("tmp_wav")    
    
    segs=[]
    # 字幕ファイルの読み込み
    for sub in subs:
        if(len(sub.content) > 0):
            wav = coeiro_old_wav(sub.content,1)
            ag = AudioSegment.from_wav(wav)
            #ag = audio_segment.read_wav_audio(wav)
            #ag = AudioSegment.from_raw(wav)
            segs.append(ag)
            
    return segs

#非推奨
def AudioSegments_to_wav(segs,subs):
    #ag1=segs[0] is AudioSegment 
    #aglast=segs[-1] is AudioSegment 
    #ag1.duration_seconds
    
    # 全音声時間を足した無音ファイルを作成
    start_time = subs[0].start.total_seconds()
    end_time = subs[-1].end.total_seconds()
    duration = (end_time - start_time)*1000
    silent_segment = AudioSegment.silent(duration=duration)
    print(duration)

    i=0
    # 音声ファイルの位置を再生ファイルの時間ごとにずらす必要がある。じゃないと全部なる笑
    for seg in segs:
        pos = subs[i].start.total_seconds()*1000
        print(pos)
        silent_segment = silent_segment.overlay(seg, position=pos)
        i+=1

    return silent_segment

#非推奨
def subtitle_to_wav(subtitle_file):
    # 字幕ファイルの読み込み
    with open(subtitle_file, mode='r', encoding="utf-8") as f: 
        subs = srt.parse(f.read()) 

        # 全音声時間を足した無音ファイルを作成
        start_time = subs[0].start.total_seconds()
        end_time = subs[-1].end.total_seconds()
        duration = (end_time - start_time)*1000
        silent_segment = AudioSegment.silent(duration=duration)
        
        for sub in subs:
            if(len(sub.content) > 0):
                wav = coeiro_old_wav(sub.content,1)
                seg = AudioSegment.from_wav(wav)
                pos = sub.start.total_seconds()*1000
                silent_segment = silent_segment.overlay(seg, position=pos)
        return silent_segment
    
    return ""

