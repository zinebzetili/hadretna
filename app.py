import streamlit as st
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import torch
import soundfile as sf

# Load the Whisper model
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")

# Load the processor
processor = WhisperProcessor.from_pretrained("openai/whisper-base")

# Function to transcribe audio
def transcribe_audio(audio_file_path):
    with sf.SoundFile(audio_file_path) as file:
        audio = file.read(dtype='float32')
        sampling_rate = file.samplerate

    # Reshape audio to match expected input format (batch_size, num_channels, sequence_length)
    audio = torch.tensor(audio).unsqueeze(0)
    mel_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt", padding="longest")

    with torch.no_grad():
        logits = model(mel_features.input_features).logits

    transcription = processor.batch_decode(logits, skip_special_tokens=True)
    return transcription

# Streamlit app
def main():
    st.title("Whisper Speech Recognition")
    audio_file = st.sidebar.file_uploader("Upload Audio File", type=["wav"])

    if audio_file is not None:
        transcriptions = transcribe_audio(audio_file)
        st.header("Transcription")
        for i, transcription in enumerate(transcriptions):
            st.write(transcription)

if __name__ == "__main__":
    main()