# Targeted Adversarial Examples for Black Box Audio Systems

Sample code to let you create your own adversarial examples! [Paper linked here](https://arxiv.org/abs/1805.07820).

## Installation
Note: linux platform required as this code uses an old version of tensorflow (1.8).

Dependencies: cuda 9.0, python 3.6, `requirements.txt`.

For example, if using anaconda (and on cuda9.0), create an environment and install the requirements:
```
conda create --name adversarialaudio python=3.6
conda activate adversarialaudio
pip install -r requirements.txt
```
Then clone the DeepSpeech repository and download the model at the appropriate version:
```
git clone -b 'v0.1.1' --single-branch --depth 1 https://github.com/mozilla/DeepSpeech.git
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.1.1/deepspeech-0.1.1-models.tar.gz
tar -xzf deepspeech-0.1.1-models.tar.gz && rm deepspeech-0.1.1-models.tar.gz
```
Finally, create the checkpoint used for the attack:
```
python make_checkpoint.py
```
DeepSpeech may throw a warning saying "decoder library file does not exist" but that can be ignored.

## Running Attacks
Now create and run an attack, for example:
```bash
python run_audio_attack.py sample_input.wav "hello world"
``` 
Of course, `sample_input.wav` may be changed to any input audio file and `"hello world"` may be changed to any target transcription.

You can also listen to pre-created audio samples in the [samples](samples/) directory. Each original/adversarial pair is denoted by a leading number, with model transcriptions as the title.
