import argparse
import os

import util.format as uf

import torch
import torchaudio

import logging
logging.basicConfig(level=logging.INFO)


class GreedyCTCDecoder(torch.nn.Module):

    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])


if __name__ == '__main__':
    # Initialize Torch
    torch.random.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get Script Argument
    parser = argparse.ArgumentParser(description="Wav2Vec2.0 ASR Inference")
    parser.add_argument("file", help="Audio File (WAV, M4A)", type=str)
    args = parser.parse_args()

    # File Conversion (if necessary) target to uf.WAV_EXT
    audio_file = args.file
    fname, ext = os.path.splitext(args.file)
    if ext == uf.M4A_EXT:
        uf.m4a_to_wav(audio_file, fname + uf.WAV_EXT)
        audio_file = fname + uf.WAV_EXT

    # Build Pipeline based on WAV2VEC2
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)

    logging.info(' Wav2Vec2.0 : WAV2VEC2_ASR_BASE_960H')
    logging.info(f' - Speech File : {audio_file}')
    logging.info(f' - Sample Rate : {bundle.sample_rate}')
    logging.info(f' - Labels : {sorted(bundle.get_labels())}')

    # Load the Audio File
    waveform, sample_rate = torchaudio.load(audio_file)
    waveform = waveform.to(device)

    if sample_rate != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

    # Infer on the Loaded Waveform
    with torch.inference_mode():
        emission, _ = model(waveform)

    # Initialize and Decode the Emission from the Waveform
    decoder = GreedyCTCDecoder(labels=bundle.get_labels())
    transcript = decoder(emission[0])

    logging.info(f' Transcript : {transcript}')
