# mindmirror

## MS1: Get (english) speech input via mic and output it as text

* Considering
  * https://github.com/myshell-ai/OpenVoice
  * https://github.com/coqui-ai/TTS
  * https://github.com/erew123/alltalk_tts/tree/alltalkbeta
  * https://github.com/SWivid/F5-TTS
  * https://github.com/speechbrain/speechbrain
  * https://github.com/openai/whisper
* Went with whisper because speechbrain gave me some dependency issues with pytorch

## MS2: Send text to ai API and print response

* Considering
  * Anthropic/claude
  * MistralAI/mixtral
  * Google/gemini
* Went with gemini as it has straight forward api and sufficient free plan
* 
* Integrated MS1 and MS2 with multiprocessing queues

## MS3: TTS with off-the-shelf voice model

* Considering
  * https://github.com/OHF-Voice/piper1-gpl
  * https://github.com/coqui-ai/TTS
* Went with PiperVoice, because easy to use with lots of voice models available
* Integrated MS1, MS2 and MS3

## MS4: Voice cloning

* Considering
  * https://github.com/coqui-ai/TTS
  * https://github.com/myshell-ai/OpenVoice
  * https://github.com/RVC-Boss/GPT-SoVITS
  * https://github.com/yl4579/StyleTTS2
* Went with StyleTTS2 because of compatibility issues with TTS and OpenVoice
  * Had issues as well
  * Then fine tune a model
    * Copy to the following files into the StyleTTS2 repo dir `/styletts2`:
      * `styletts2/data/wavs_clean`
      * `styletts2/data/config_ft.yml`
      * `styletts2/data/val_list.txt`
      * `styletts2/data/train_list.txt`
    * Adjust the paths inside `config_ft.yml`
       * ```yml
         data_params:
           root_path: "xxx/StyleTTS2/styletts2/data/wavs_clean"
           train_data: "xxx/StyleTTS2/styletts2/train_list.txt"
           val_data: "xxx/StyleTTS2/styletts2/val_list.txt"
         ```
    * Cd into xxx/StyleTTS2
      * `python train_finetune.py --config_path styletts2/config_ft.yml`
      * Just no way to get this to work with my hardware
* Trying with GPT-SoVITS
  * Failed again
* Trying with F5-TTS
  1. Install F5-TTS repository: clone and then `(mindmirror) repos/mindmirror$ ` `pip install -e .` 
  2. Record voice sample: `(mindmirror) repos/mindmirror$ ` `python record_sample.py`
  3. Adjust to the pretrained vocab `PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")` in `repos/F5-TTS/src/f5_tts/train/datasets/prepare_csv_wavs.py`
  4. Prepare dataset: `(mindmirror) repos/F5-TTS$ ` `python src/f5_tts/train/datasets/prepare_csv_wavs.py ../mindmirror/data/MyVoice/ ./data/MyVoice_pinyin`
  5. Adjust `num_workers=os.cpu_count()` in `repos/F5-TTS/src/f5_tts/train/finetune_cli.py`
  6. Train on top of model F5TTS v1 base
    * `(mindmirror) repos/F5-TTS$ ` `python src/f5_tts/train/finetune_cli.py --exp_name F5TTS_v1_Base --dataset_name MyVoice --finetune --pretrain ckpts/F5TTS_v1_Base/model_1250000.safetensors --tokenizer pinyin --learning_rate 5e-5 --epochs 50 --batch_size_type sample --batch_size_per_gpu 1 --grad_accumulation_steps 4     --save_per_updates 5000 --keep_last_n_checkpoints 1`
  7. Test loading the model `(mindmirror) repos/mindmirror$ ` `python f5_tts/verify_model.py`
  8. Test inference
   * `(mindmirror) repos/F5-TTS$` `python src/f5_tts/infer/infer_cli.py --model F5TTS_v1_Base --ckpt_file ckpts/MyVoice/model_last.pt --ref_audio ../mindmirror/voice_samples/wavs_clean/paragraph_01.wav --ref_text "The birch canoe slid on the smooth planks." --gen_text "This is a test. I am checking if my voice model is overtrained."`
* `(mindmirror) repos/F5-TTS$` `python integration.py`


# How to run

1. Create conda env
   1. `repos/mindmirror$ ` `conda create -n mindmirror python=3.11 -y`
   2. `repos/mindmirror$ ``conda activate mindmirror`
   3. `(mindmirror) repos/mindmirror$ ` `pip install -r requirements.txt`
2. Add huggingface api key and gemini api key to .env
3. download pipervoice of your choice into `repos/mindmirror/pipervoice/en/` (I used semaine)
4. Edit integration.py to use piper as tts system
5. Run `(mindmirror) repos/mindmirror$ ` `python integration.py`

For custom voice:
1. Clone and install F5-TTS repository: clone and then `(mindmirror) repos/mindmirror$ ` `pip install -e .`
2. Record voice sample: `(mindmirror) repos/mindmirror$ ` `python record_sample.py`
3. Download the pretrained model "F5TTS_v1_Base" from huggingface
4. Adjust to the pretrained vocab `PRETRAINED_VOCAB_PATH = files("f5_tts").joinpath("../../data/Emilia_ZH_EN_pinyin/vocab.txt")` in `repos/F5-TTS/src/f5_tts/train/datasets/prepare_csv_wavs.py`
5. Prepare dataset: `(mindmirror) repos/F5-TTS$ ` `python src/f5_tts/train/datasets/prepare_csv_wavs.py ../mindmirror/data/MyVoice/ ./data/MyVoice_pinyin`
6. Adjust `num_workers=os.cpu_count()` in `repos/F5-TTS/src/f5_tts/train/finetune_cli.py`
7. Train on top of model F5TTS v1 base
   * `(mindmirror) repos/F5-TTS$ ` `python src/f5_tts/train/finetune_cli.py --exp_name F5TTS_v1_Base --dataset_name MyVoice --finetune --pretrain ckpts/F5TTS_v1_Base/model_1250000.safetensors --tokenizer pinyin --learning_rate 5e-5 --epochs 50 --batch_size_type sample --batch_size_per_gpu 1 --grad_accumulation_steps 4     --save_per_updates 5000 --keep_last_n_checkpoints 1`
8. Test loading the model `(mindmirror) repos/mindmirror$ ` `python f5_tts/verify_model.py`
9. Test inference
   * `(mindmirror) repos/F5-TTS$` `python src/f5_tts/infer/infer_cli.py --model F5TTS_v1_Base --ckpt_file ckpts/MyVoice/model_last.pt --ref_audio ../mindmirror/voice_samples/wavs_clean/paragraph_01.wav --ref_text "The birch canoe slid on the smooth planks." --gen_text "This is a test. I am checking if my voice model is overtrained."`
10. Run `(mindmirror) repos/mindmirror$ ` `python integration.py`