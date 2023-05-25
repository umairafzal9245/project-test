import time
from typing import List

import numpy as np
# import pysbd
import torch

from TTS.config import load_config
from TTS.tts.models import setup_model as setup_tts_model

# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
from TTS.tts.utils.synthesis import synthesis, transfer_voice, trim_silence
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.models import setup_model as setup_vocoder_model
from TTS.vocoder.utils.generic_utils import interpolate_vocoder_input
from scipy.io.wavfile import write


class Synthesizer(object):
    def __init__(
        self,
        tts_checkpoint: str,
        tts_config_path: str,
        tts_speakers_file: str = "",
        tts_languages_file: str = "",
        vocoder_checkpoint: str = "",
        vocoder_config: str = "",
        encoder_checkpoint: str = "",
        encoder_config: str = "",
        use_cuda: bool = False,
    ) -> None:
        """General 🐸 TTS interface for inference. It takes a tts and a vocoder
        model and synthesize speech from the provided text.

        The text is divided into a list of sentences using `pysbd` and synthesize
        speech on each sentence separately.

        If you have certain special characters in your text, you need to handle
        them before providing the text to Synthesizer.

        TODO: set the segmenter based on the source language

        Args:
            tts_checkpoint (str): path to the tts model file.
            tts_config_path (str): path to the tts config file.
            vocoder_checkpoint (str, optional): path to the vocoder model file. Defaults to None.
            vocoder_config (str, optional): path to the vocoder config file. Defaults to None.
            encoder_checkpoint (str, optional): path to the speaker encoder model file. Defaults to `""`,
            encoder_config (str, optional): path to the speaker encoder config file. Defaults to `""`,
            use_cuda (bool, optional): enable/disable cuda. Defaults to False.
        """
        self.tts_checkpoint = tts_checkpoint
        self.tts_config_path = tts_config_path
        self.tts_speakers_file = tts_speakers_file
        self.tts_languages_file = tts_languages_file
        self.vocoder_checkpoint = vocoder_checkpoint
        self.vocoder_config = vocoder_config
        self.encoder_checkpoint = encoder_checkpoint
        self.encoder_config = encoder_config
        self.use_cuda = use_cuda

        self.tts_model = None
        self.vocoder_model = None
        self.speaker_manager = None
        self.tts_speakers = {}
        self.language_manager = None
        self.num_languages = 0
        self.tts_languages = {}
        self.d_vector_dim = 0
        self.use_cuda = use_cuda

        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."
        self._load_tts(tts_checkpoint, tts_config_path, use_cuda)
        self.output_sample_rate = self.tts_config.audio["sample_rate"]
        if vocoder_checkpoint:
            self._load_vocoder(vocoder_checkpoint, vocoder_config, use_cuda)
            self.output_sample_rate = self.vocoder_config.audio["sample_rate"]

    def _load_tts(self, tts_checkpoint: str, tts_config_path: str, use_cuda: bool) -> None:
        """Load the TTS model.

        1. Load the model config.
        2. Init the model from the config.
        3. Load the model weights.
        4. Move the model to the GPU if CUDA is enabled.
        5. Init the speaker manager in the model.

        Args:
            tts_checkpoint (str): path to the model checkpoint.
            tts_config_path (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        # pylint: disable=global-statement
        self.tts_config = load_config(tts_config_path)
        if self.tts_config["use_phonemes"] and self.tts_config["phonemizer"] is None:
            raise ValueError("Phonemizer is not defined in the TTS config.")

        self.tts_model = setup_tts_model(config=self.tts_config)

        if not self.encoder_checkpoint:
            self._set_speaker_encoder_paths_from_tts_config()

        self.tts_model.load_checkpoint(self.tts_config, tts_checkpoint, eval=True)
        if use_cuda:
            self.tts_model.cuda()

        if self.encoder_checkpoint and hasattr(self.tts_model, "speaker_manager"):
            self.tts_model.speaker_manager.init_encoder(self.encoder_checkpoint, self.encoder_config, use_cuda)

    def _set_speaker_encoder_paths_from_tts_config(self):
        """Set the encoder paths from the tts model config for models with speaker encoders."""
        if hasattr(self.tts_config, "model_args") and hasattr(
            self.tts_config.model_args, "speaker_encoder_config_path"
        ):
            self.encoder_checkpoint = self.tts_config.model_args.speaker_encoder_model_path
            self.encoder_config = self.tts_config.model_args.speaker_encoder_config_path

    def _load_vocoder(self, model_file: str, model_config: str, use_cuda: bool) -> None:
        """Load the vocoder model.

        1. Load the vocoder config.
        2. Init the AudioProcessor for the vocoder.
        3. Init the vocoder model from the config.
        4. Move the model to the GPU if CUDA is enabled.

        Args:
            model_file (str): path to the model checkpoint.
            model_config (str): path to the model config file.
            use_cuda (bool): enable/disable CUDA use.
        """
        self.vocoder_config = load_config(model_config)
        self.vocoder_ap = AudioProcessor(verbose=False, **self.vocoder_config.audio)
        self.vocoder_model = setup_vocoder_model(self.vocoder_config)
        self.vocoder_model.load_checkpoint(self.vocoder_config, model_file, eval=True)
        if use_cuda:
            self.vocoder_model.cuda()


    def save_wav(self, wav: List[int], path: str) -> None:
        """Save the waveform as a file.

        Args:
            wav (List[int]): waveform as a list of values.
            path (str): output path to save the waveform.
        """
        wav = np.array(wav)
        self.tts_model.ap.save_wav(wav, path, self.output_sample_rate)

    # def get_embedding(self, wav) -> np.ndarray:
    #     """Get the embedding from the speaker encoder.

    #     Args:
    #         wav (List[int]): waveform as a list of values.

    #     Returns:
    #         np.ndarray: embedding vector.
    #     """
    #     if not self.tts_model.speaker_manager:
    #         raise ValueError("Speaker encoder is not defined in the TTS model.")
      
    #     return self.tts_model.speaker_manager.compute_embedding_from_clip(wav)

    def tts(
        self,
        text: str = "",
        speaker_name=None,
        speaker_wav=None,
    ) -> List[int]:
       

        if not text:
            raise ValueError(
                "You need to define either `text` (for sythesis)"
            )

        if text:
            print("text: ", text)

        # handle multi-speaker
        speaker_embedding = None
        speaker_id = None
        if self.tts_speakers_file or hasattr(self.tts_model.speaker_manager, "name_to_id"):

            # handle Neon models with single speaker.
            if len(self.tts_model.speaker_manager.name_to_id) == 1:
                speaker_id = list(self.tts_model.speaker_manager.name_to_id.values())[0]

            elif speaker_name and isinstance(speaker_name, str):
                # with open("spea.txt", "w") as f:
                #         f.write(str(list(self.tts_model.speaker_manager.name_to_id.keys())))
                if self.tts_config.use_d_vector_file:
                    # get the average speaker embedding from the saved d_vectors.
                    speaker_embedding = self.tts_model.speaker_manager.get_mean_embedding(
                        speaker_name, num_samples=None, randomize=False
                    )
                    speaker_embedding = np.array(speaker_embedding)[None, :]  # [1 x embedding_dim]
                else:
                    # get speaker idx from the speaker name
                    speaker_id = self.tts_model.speaker_manager.name_to_id[speaker_name]
                    # print all speaker names
                    
                    
                    

            elif not speaker_name and not speaker_wav:
                raise ValueError(
                    " [!] Look like you use a multi-speaker model. "
                    "You need to define either a `speaker_name` or a `speaker_embeddings` to use a multi-speaker model."
                )
            else:
                speaker_embedding = None
        else:
            if speaker_name:
                raise ValueError(
                    f" [!] Missing speakers.json file path for selecting speaker {speaker_name}."
                    "Define path for speaker.json if it is a multi-speaker model or remove defined speaker idx. "
                )

        # compute a new d_vector from the given clip.
        if speaker_wav is not None:
            speaker_embedding = self.tts_model.speaker_manager.compute_embedding_from_clip(speaker_wav)

        use_gl = self.vocoder_model is None

            # synthesize voice
        outputs = synthesis(
                model=self.tts_model,
                text=text,
                CONFIG=self.tts_config,
                use_cuda=self.use_cuda,
                speaker_id=speaker_id,
                style_wav=None,
                style_text=None,
                use_griffin_lim=use_gl,
                d_vector=speaker_embedding,
                language_id=None,
        )
       
        waveform = outputs["wav"]
        mel_postnet_spec = outputs["outputs"]["model_outputs"][0].detach().cpu().numpy()
        if not use_gl:
            # denormalize tts output based on tts audio config
            mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
            device_type = "cuda" if self.use_cuda else "cpu"
            # renormalize spectrogram based on vocoder config
            vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
            # compute scale factor for possible sample rate mismatch
            scale_factor = [
                1,
                    self.vocoder_config["audio"]["sample_rate"] / self.tts_model.ap.sample_rate,
                ]
            if scale_factor[1] != 1:
                print(" > interpolating tts model output.")
                vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
            else:
                vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                # run vocoder model
                # [1, T, C]
            
            waveform = self.vocoder_model.inference(vocoder_input.to(device_type))
            
        if self.use_cuda and not use_gl:
            waveform = waveform.cpu()
        if not use_gl:
            waveform = waveform.numpy()
            
        waveform = waveform.squeeze()

            # trim silence
        if "do_trim_silence" in self.tts_config.audio and self.tts_config.audio["do_trim_silence"]:
            waveform = trim_silence(waveform, self.tts_model.ap)

        return waveform
    
    def removesilence(self, wav):
        wav = trim_silence(wav, self.tts_model.ap)
        return wav
    
    def voice_conversion(self, speaker_wav=None,referencewav=None):

        speaker_embedding = None
        if speaker_wav is not None:
            speaker_embedding = self.tts_model.speaker_manager.compute_embedding_from_clip(speaker_wav)

        reference_embedding = None
        if referencewav is not None:
            reference_embedding = self.tts_model.speaker_manager.compute_embedding_from_clip(referencewav)

        use_gl = self.vocoder_model is None

            # synthesize voice
        outputs = transfer_voice(
                model=self.tts_model,
                CONFIG=self.tts_config,
                use_cuda=self.use_cuda,
                reference_wav=referencewav,
                d_vector=speaker_embedding,
                reference_d_vector=reference_embedding,
                use_griffin_lim=use_gl,
        )

        waveform = outputs
       
        if not use_gl:
            mel_postnet_spec = outputs[0].detach().cpu().numpy()
            # denormalize tts output based on tts audio config
            mel_postnet_spec = self.tts_model.ap.denormalize(mel_postnet_spec.T).T
            device_type = "cuda" if self.use_cuda else "cpu"
            # renormalize spectrogram based on vocoder config
            vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
            # compute scale factor for possible sample rate mismatch
            scale_factor = [
                1,
                    self.vocoder_config["audio"]["sample_rate"] / self.tts_model.ap.sample_rate,
                ]
            if scale_factor[1] != 1:
                print(" > interpolating tts model output.")
                vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
            else:
                vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                # run vocoder model
                # [1, T, C]
            
            waveform = self.vocoder_model.inference(vocoder_input.to(device_type))
            
        if self.use_cuda and not use_gl:
            waveform = waveform.cpu()
        if not use_gl:
            waveform = waveform.numpy()
            
        waveform = waveform.squeeze()

        return waveform
