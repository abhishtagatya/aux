from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

if __name__ == '__main__':
    # load model and processor
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h-lv60-self",)
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h-lv60-self")

    # load dummy lib and read soundfiles
    ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")

    print(ds[1])

    # tokenize
    input_values = processor(
        ds[1]["audio"]["array"],
        return_tensors="pt",
        padding="longest",
        sampling_rate=16000,
    ).input_values

    # retrieve logits
    feature = model(input_values)
    print(feature)

    # take argmax and decode
    predicted_ids = torch.argmax(feature.logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    print(transcription)
