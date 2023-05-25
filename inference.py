from TTS.utils.synthesizer import Synthesizer
from phonemizer import phonemize
import numpy as np

synthesizer = Synthesizer(
        tts_checkpoint="./model/best_model.pth",
        tts_config_path="./model/config.json",
        tts_speakers_file="./model/speakers.pth",
        encoder_checkpoint="./speaker_encoder/model_se.pth.tar",
        encoder_config="./speaker_encoder/config_se.json",
        use_cuda=False)


def synthesize(text, name=None, wavfilee=None):
    text = phonemize(text, language='ur', backend='espeak', strip=True, preserve_punctuation=False,language_switch='remove-flags',with_stress=True)
    text += "."   
    audio = synthesizer.tts(text=text,speaker_name=name,speaker_wav=wavfilee)
    return audio


if __name__=="__main__":

    audio = synthesize("میں ایک اچھا لڑکا ہوں", name="VCTK_old_speaker1")
    synthesizer.save_wav(audio, "./audio.wav")