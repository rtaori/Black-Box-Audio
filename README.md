# Targeted Adversarial Examples for Black Box Audio Systems

Sample code to let you create your own adversarial examples!

Setup:
```bash
pip3 install -r requirements.txt
git clone https://github.com/mozilla/DeepSpeech.git
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.1.0/deepspeech-0.1.0-models.tar.gz
tar -xzf deepspeech-0.1.0-models.tar.gz
rm deepspeech-0.1.0-models.tar.gz
python3 make_checkpoint.py
```

Now create run an attack with the format:
```bash
python3 run_audio_attack.py input_file.wav "target phrase"
```
For example,
```bash
python3 run_audio_attack.py sample_input.wav "hello world"
``` 

You can also listen to pre-created audio samples in the [samples](samples/) directory. Each original/adversarial pair is denoted by a leading number, with model transcriptions as the title.
