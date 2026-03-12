import time
import torch
import torchaudio

from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM

device = "cuda:1"
encoder = Encoder.from_pretrained("/data/models/tada-codec", subfolder="encoder", language="ch").to(device)
model = TadaForCausalLM.from_pretrained("/data/models/tada-3b-ml").to(device)

# Load a reference audio clip in the target language
audio, sample_rate = torchaudio.load("/data/input/shot4-prompt.wav")
audio = audio.to(device)

# For non-English prompts, provide the transcript so the encoder uses forced alignment
# instead of the built-in ASR (which is English-only)
prompt_text = "从武昌三佛阁到东厂口，再到珞珈山，这已经是武汉大学历史上的第三个校址。也就是在这个新校址上，建成了中国高校历史上最宏伟壮观的大学校园建筑群，成就了武汉大学全国最美校园的赞誉，也塑造了武汉大学近现代教育史上最辉煌的时代篇章。"
prompt = encoder(audio, text=[prompt_text], sample_rate=sample_rate)


start_time = time.time()

output = model.generate(
    prompt=prompt,
    text="各位学员大家好，今天我们来开始武汉大学红船精神现场教学课程。首先我们来认识武汉大学珞珈山校园的近代建筑群，感悟位列民国四大名校的武汉大学的历史文化。1928年国立武汉大学成立后，选定在武昌东湖边的珞珈山建设新校舍。",
)
elapsed_time = time.time() - start_time
print(f"Model generation took {elapsed_time:.2f} seconds")

# Save the output audio to a file
if output.audio is not None and len(output.audio) > 0:
    # GenerationOutput.audio is a list of tensors
    audio_output = output.audio[0]  # Get the first audio tensor
    sample_rate = 24000  # Default sample rate for TTS models
    torchaudio.save("output.wav", audio_output.cpu(), sample_rate)
    print(f"Audio saved to output.wav")