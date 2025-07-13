<p align="center">
<a href="https://github.com/nari-labs/dia">
<img src="./dia/static/images/banner.png">
</a>
</p>
<p align="center">
<a href="https://tally.so/r/meokbo" target="_blank"><img alt="Static Badge" src="https://img.shields.io/badge/Join-Waitlist-white?style=for-the-badge"></a>
<a href="https://discord.gg/bJq6vjRRKv" target="_blank"><img src="https://img.shields.io/badge/Discord-Join%20Chat-7289DA?logo=discord&style=for-the-badge"></a>
<a href="https://github.com/nari-labs/dia/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge" alt="LICENSE"></a>
</p>
<p align="center">
<a href="https://huggingface.co/nari-labs/Dia-1.6B-0626"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg" alt="Model on HuggingFace" height=42 ></a>
<a href="https://huggingface.co/spaces/nari-labs/Dia-1.6B"><img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-lg-dark.svg" alt="Space on HuggingFace" height=38></a>
</p>

Dia is a 1.6B parameter text to speech model created by Nari Labs.

**UPDATE 🤗(06/27)**: Dia is now available through [Hugging Face Transformers](https://github.com/huggingface/transformers)!

Dia **directly generates highly realistic dialogue from a transcript**. You can condition the output on audio, enabling emotion and tone control. The model can also produce nonverbal communications like laughter, coughing, clearing throat, etc.

To accelerate research, we are providing access to pretrained model checkpoints and inference code. The model weights are hosted on [Hugging Face](https://huggingface.co/nari-labs/Dia-1.6B-0626). The model only supports English generation at the moment.

We also provide a [demo page](https://yummy-fir-7a4.notion.site/dia) comparing our model to [ElevenLabs Studio](https://elevenlabs.io/studio) and [Sesame CSM-1B](https://github.com/SesameAILabs/csm).

- We have a ZeroGPU Space running! Try it now [here](https://huggingface.co/spaces/nari-labs/Dia-1.6B-0626). Thanks to the HF team for the support :)
- Join our [discord server](https://discord.gg/bJq6vjRRKv) for community support and access to new features.
- Play with a larger version of Dia: generate fun conversations, remix content, and share with friends. 🔮 Join the [waitlist](https://tally.so/r/meokbo) for early access.

## Generation Guidelines

- Keep input text length moderate 
    - Short input (corresponding to under 5s of audio) will sound unnatural
    - Very long input (corresponding to over 20s of audio) will make the speech unnaturally fast.
- Use non-verbal tags sparingly, from the list in the README. Overusing or using unlisted non-verbals may cause weird artifacts.
- Always begin input text with `[S1]`, and always alternate between `[S1]` and `[S2]` (i.e. `[S1]`... `[S1]`... is not good)
- When using audio prompts (voice cloning), follow these instructions carefully:
    - Provide the transcript of the to-be cloned audio before the generation text.
    - Transcript must use `[S1]`, `[S2]` speaker tags correctly (i.e. single speaker: `[S1]`..., two speakers: `[S1]`... `[S2]`...)
    - Duration of the to-be cloned audio should be 5~10 seconds for the best results.
        (Keep in mind: 1 second ≈ 86 tokens)
- Put `[S1]` or `[S2]` (the second-to-last speaker's tag) at the end of the audio to improve audio quality at the end

## Quickstart

### Transformers Support

We now have a [Hugging Face Transformers](https://github.com/huggingface/transformers) implementation of Dia! You should install `main` branch of `transformers` to use it. See [hf.py](hf.py) for more information.

<details>
<summary>View more details</summary>

Install `main` branch of `transformers`

```bash
pip install git+https://github.com/huggingface/transformers.git
# or install with uv
uv pip install git+https://github.com/huggingface/transformers.git
```

Run `hf.py`. The file is as below.

```python
from transformers import AutoProcessor, DiaForConditionalGeneration


torch_device = "cuda"
model_checkpoint = "nari-labs/Dia-1.6B-0626"

text = [
    "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
]
processor = AutoProcessor.from_pretrained(model_checkpoint)
inputs = processor(text=text, padding=True, return_tensors="pt").to(torch_device)

model = DiaForConditionalGeneration.from_pretrained(model_checkpoint).to(torch_device)
outputs = model.generate(
    **inputs, max_new_tokens=3072, guidance_scale=3.0, temperature=1.8, top_p=0.90, top_k=45
)

outputs = processor.batch_decode(outputs)
processor.save_audio(outputs, "example.mp3")
```

</details>

### Run with this repo

<details>
<summary> Install via pip </summary>

```bash
# Clone this repository
git clone https://github.com/nari-labs/dia.git
cd dia

# Optionally
python -m venv .venv && source .venv/bin/activate

# Install dia
pip install -e .
```

Or you can install without cloning.

```bash
# Install directly from GitHub
pip install git+https://github.com/nari-labs/dia.git
```

Now, run some examples.

```bash
python example/simple.py
```
</details>


<details>
<summary>Install via uv</summary>

You need [uv](https://docs.astral.sh/uv/) to be installed.

```bash
# Clone this repository
git clone https://github.com/nari-labs/dia.git
cd dia
```

Run some examples directly.

```bash
uv run example/simple.py
```

</details>

<details>
<summary>Run Gradio UI</summary>

```bash
python app.py

# Or if you have uv installed
uv run app.py
```

</details>

<details>
<summary>Run with CLI</summary>

```bash
python cli.py --help

# Or if you have uv installed
uv run cli.py --help
```

</details>

> [!NOTE]
> The model was not fine-tuned on a specific voice. Hence, you will get different voices every time you run the model.
> You can keep speaker consistency by either adding an audio prompt, or fixing the seed.

> [!IMPORTANT]
> If you are using 5000 series GPU, you should use torch 2.8 nightly. Look at the issue [#26](https://github.com/nari-labs/dia/issues/26) for more details.

## Features

- Generate dialogue via `[S1]` and `[S2]` tag
- Generate non-verbal like `(laughs)`, `(coughs)`, etc.
  - Below verbal tags will be recognized, but might result in unexpected output.
  - `(laughs), (clears throat), (sighs), (gasps), (coughs), (singing), (sings), (mumbles), (beep), (groans), (sniffs), (claps), (screams), (inhales), (exhales), (applause), (burps), (humming), (sneezes), (chuckle), (whistles)`
- Voice cloning. See [`example/voice_clone.py`](example/voice_clone.py) for more information.
  - In the Hugging Face space, you can upload the audio you want to clone and place its transcript before your script. Make sure the transcript follows the required format. The model will then output only the content of your script.


## 💻 Hardware and Inference Speed

Dia has been tested on only GPUs (pytorch 2.0+, CUDA 12.6). CPU support is to be added soon.
The initial run will take longer as the Descript Audio Codec also needs to be downloaded.

These are the speed we benchmarked in RTX 4090.

| precision | realtime factor w/ compile | realtime factor w/o compile | VRAM |
|:-:|:-:|:-:|:-:|
| `bfloat16` | x2.1 | x1.5 | ~4.4GB |
| `float16` | x2.2 | x1.3 | ~4.4GB |
| `float32` | x1 | x0.9 | ~7.9GB |

We will be adding a quantized version in the future.

If you don't have hardware available or if you want to play with bigger versions of our models, join the waitlist [here](https://tally.so/r/meokbo).

## 🪪 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This project offers a high-fidelity speech generation model intended for research and educational use. The following uses are **strictly forbidden**:

- **Identity Misuse**: Do not produce audio resembling real individuals without permission.
- **Deceptive Content**: Do not use this model to generate misleading content (e.g. fake news)
- **Illegal or Malicious Use**: Do not use this model for activities that are illegal or intended to cause harm.

By using this model, you agree to uphold relevant legal standards and ethical responsibilities. We **are not responsible** for any misuse and firmly oppose any unethical usage of this technology.

## 🔭 TODO / Future Work

- Docker support for ARM architecture and MacOS.
- Optimize inference speed.
- Add quantization for memory efficiency.

## 🤝 Contributing

We are a tiny team of 1 full-time and 1 part-time research-engineers. We are extra-welcome to any contributions!
Join our [Discord Server](https://discord.gg/bJq6vjRRKv) for discussions.

## 🤗 Acknowledgements

- We thank the [Google TPU Research Cloud program](https://sites.research.google/trc/about/) for providing computation resources.
- Our work was heavily inspired by [SoundStorm](https://arxiv.org/abs/2305.09636), [Parakeet](https://jordandarefsky.com/blog/2024/parakeet/), and [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).
- Hugging Face for providing the ZeroGPU Grant.
- "Nari" is a pure Korean word for lily.
- We thank Jason Y. for providing help with data filtering.


## ⭐ Star History

<a href="https://www.star-history.com/#nari-labs/dia&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=nari-labs/dia&type=Date" />
 </picture>
</a>
<!-- from dia.model import Dia
example code: -->

<!-- model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")

# You should put the transcript of the voice you want to clone
# We will use the audio created by running simple.py as an example.
# Note that you will be REQUIRED TO RUN simple.py for the script to work as-is.
clone_from_text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
clone_from_audio = "simple.mp3"

# For your custom needs, replace above with below and add your audio file to this directory:
# clone_from_text = "[S1] ... [S2] ... [S1] ... corresponding to your_audio_name.mp3"
# clone_from_audio = "your_audio_name.mp3"

# Text to generate
text_to_generate = "[S1] Hello, how are you? [S2] I'm good, thank you. [S1] What's your name? [S2] My name is Dia. [S1] Nice to meet you. [S2] Nice to meet you too."

# It will only return the audio from the text_to_generate
output = model.generate(
    clone_from_text + text_to_generate,
    audio_prompt=clone_from_audio,
    use_torch_compile=False,
    verbose=True,
    cfg_scale=4.0,
    temperature=1.8,
    top_p=0.90,
    cfg_filter_top_k=50,
)

model.save_audio("voice_clone.mp3", output)

from dia.model import Dia


model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")

text = "[S1] Dia is an open weights text to dialogue model. [S2] You get full control over scripts and voices. [S1] Wow. Amazing. (laughs) [S2] Try it now on Git hub or Hugging Face."
texts = [text for _ in range(10)]

output = model.generate(texts, use_torch_compile=True, verbose=True, max_tokens=1500)

for i, o in enumerate(output):
    model.save_audio(f"simple_{i}.mp3", o) -->





# @title
import time
import torch
import soundfile as sf
import ipywidgets as widgets
from IPython.display import display, clear_output
import IPython.display as ipd
import numpy as np
from dia.model import Dia
import os
import glob
import re
import json
import threading

# Animation control variables
animation_running = False
animation_thread = None

# Default example text - will be replaced by loaded text files
example_text = """[S1] Welcome back to another episode of AI Unfiltered! I'm Jamie.
[S2] And I'm Taylor. Today, we have some really exciting news from the text-to-speech frontier."""

# Default example transcript for voice cloning
example_transcript = """[S1] This is an example of a voice cloning transcript.
[S2] Make sure this transcript matches exactly what's said in your audio file."""

def animate_loading_emojis(output_widget, base_message):
    """Animate rotating emojis while generation is running"""
    global animation_running
    emojis = ["🎤", "🔊", "🎵"]  # microphone, speaker, musical note
    emoji_index = 0

    while animation_running:
        with output_widget:
            clear_output(wait=True)
            current_emoji = emojis[emoji_index % len(emojis)]
            print(f"{base_message} {current_emoji}")

        emoji_index += 1
        time.sleep(0.6)  # Rotate every 0.6 seconds

def start_loading_animation(output_widget, message):
    """Start the loading animation"""
    global animation_running, animation_thread
    animation_running = True
    animation_thread = threading.Thread(target=animate_loading_emojis, args=(output_widget, message))
    animation_thread.daemon = True
    animation_thread.start()

def stop_loading_animation():
    """Stop the loading animation"""
    global animation_running
    animation_running = False
    if animation_thread:
        animation_thread.join(timeout=1)

def save_clone_config(clone_file, transcript_source, transcript_text="", cfg_scale=3.0, speed_factor=0.92):
    """Save the current voice cloning configuration to clone_config.txt"""
    config = {
        "clone_file": clone_file,
        "transcript_source": transcript_source,
        "transcript_text": transcript_text,
        "cfg_scale": cfg_scale,
        "speed_factor": speed_factor,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    try:
        with open("/content/clone_config.txt", "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Clone configuration saved: {clone_file} with {transcript_source}")
        print(f"✅ Audio parameters saved: CFG Scale={cfg_scale}, Speed Factor={speed_factor}")
        return True
    except Exception as e:
        print(f"❌ Error saving clone config: {e}")
        return False

def load_clone_config():
    """Load the voice cloning configuration from clone_config.txt"""
    try:
        if os.path.exists("/content/clone_config.txt"):
            with open("/content/clone_config.txt", "r") as f:
                config = json.load(f)
            return config
        else:
            return None
    except Exception as e:
        print(f"❌ Error loading clone config: {e}")
        return None

def natural_sort_key(s):
    """
    Sort strings with numbers in natural order (e.g., part_1.txt, part_2.txt, ..., part_10.txt)
    instead of lexicographical order (part_1.txt, part_10.txt, part_2.txt, ...)
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def ensure_directories():
    """Create the necessary directories if they don't exist"""
    os.makedirs("/content/texts", exist_ok=True)
    os.makedirs("/content/recordings", exist_ok=True)
    os.makedirs("/content/clone", exist_ok=True)

    # If no text files exist in /content/texts, create a sample file
    if not glob.glob("/content/texts/*.txt"):
        with open("/content/texts/part_1.txt", "w") as f:
            f.write(example_text)

def get_text_files():
    """Get a sorted list of .txt files from the /content/texts directory"""
    files = glob.glob("/content/texts/*.txt")
    return sorted(files, key=natural_sort_key)

def get_clone_audio_files():
    """Get a sorted list of audio files from the /content/clone directory"""
    mp3_files = glob.glob("/content/clone/*.mp3")
    wav_files = glob.glob("/content/clone/*.wav")
    files = mp3_files + wav_files
    return sorted(files, key=natural_sort_key)

def get_transcript_for_audio(audio_path):
    """
    Attempt to find and load a transcript file for the given audio file.
    The transcript file should have the same name as the audio file but with .txt extension.

    Args:
        audio_path: Path to the audio file

    Returns:
        The transcript content if found, None otherwise
    """
    base_name = os.path.splitext(audio_path)[0]
    transcript_path = f"{base_name}.txt"

    if os.path.exists(transcript_path):
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading transcript file {transcript_path}: {e}")
            return None
    return None

def load_text_from_file(file_path):
    """Load text content from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

def generate_audio(
    text,
    output_path,
    output_format="wav",
    max_tokens=3072,
    cfg_scale=3.0,
    temperature=1.2,
    top_p=0.95,
    cfg_filter_top_k=45,
    speed_factor=0.92,
    seed=None,
    compute_dtype="float16",
    use_torch_compile=True,
    use_voice_clone=False,
    clone_audio_path=None,
    clone_transcript=None
):
    """
    Generate speech using the Dia model with customizable parameters.

    Args:
        text: Input text to convert to speech
        output_path: Path to save the output audio file
        output_format: Format to save the audio file (wav or mp3)
        max_tokens: Maximum number of tokens to generate (default 3072 ~ 36 seconds)
        cfg_scale: Guidance scale to adhere to the text prompt (default 3.0)
        temperature: Randomness of generation (default 1.2)
        top_p: Nucleus sampling threshold (default 0.95)
        cfg_filter_top_k: Number of top logits for CFG filtering (default 45)
        speed_factor: Speed adjustment for audio output (default 0.94)
        seed: Random seed for reproducibility (default None)
        compute_dtype: Computation precision (default "float16")
        use_torch_compile: Whether to use torch.compile (default True)
        use_voice_clone: Whether to use voice cloning (default False)
        clone_audio_path: Path to the audio file to clone (default None)
        clone_transcript: Transcript of the audio to clone (default None)

    Returns:
        Path to the generated audio file
    """
    # Ensure numpy is imported in this scope
    import numpy as np

    # Check if model exists, if not, load it
    global model
    if 'model' not in globals():
        model = Dia.from_pretrained("nari-labs/Dia-1.6B", compute_dtype=compute_dtype)

    start_time = time.time()

    # Set seed if provided - implemented according to Dia's official method
    if seed is not None:
        # Set all necessary random seeds for full determinism
        import random
        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for cuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Generate audio with parameters
    if use_voice_clone and clone_audio_path and clone_transcript:
        # Voice cloning mode
        combined_text = f"{clone_transcript}\n{text}"
        output = model.generate(
            text=combined_text,
            audio_prompt=clone_audio_path,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            use_torch_compile=use_torch_compile,
            verbose=False  # Set to False to reduce output during batch processing
        )
    else:
        # Standard generation mode
        output = model.generate(
            text=text,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            use_torch_compile=use_torch_compile,
            verbose=False  # Set to False to reduce output during batch processing
        )

    # Apply speed factor if different from 1.0
    if speed_factor != 1.0:
        new_length = int(len(output) / speed_factor)
        indices = np.linspace(0, len(output) - 1, new_length)
        output = np.interp(indices, np.arange(len(output)), output)

    # Ensure the output path has the correct extension
    base_output_path = os.path.splitext(output_path)[0]
    temp_wav_file = f"{base_output_path}.wav"

    # Save the audio file as WAV (required for MP3 conversion)
    sf.write(temp_wav_file, output, 44100)

    # If MP3 is selected, convert the WAV file to MP3
    final_output_file = temp_wav_file
    if output_format == "mp3":
        try:
            from pydub import AudioSegment
            final_output_file = f"{base_output_path}.mp3"
            AudioSegment.from_wav(temp_wav_file).export(final_output_file, format="mp3")
            # Remove the temporary WAV file if MP3 conversion was successful
            if os.path.exists(final_output_file):
                os.remove(temp_wav_file)
        except ImportError:
            final_output_file = temp_wav_file
        except Exception as e:
            final_output_file = temp_wav_file

    return final_output_file

def batch_process_files(
    output_format,
    max_tokens,
    cfg_scale,
    temperature,
    top_p,
    cfg_filter_top_k,
    speed_factor,
    seed,
    compute_dtype,
    use_torch_compile,
    use_voice_clone,
    clone_audio_path,
    clone_transcript,
    output_area
):
    """Process all text files in the /content/texts directory"""
    with output_area:
        clear_output()

        # Get all text files
        text_files = get_text_files()

        if not text_files:
            print("No text files found in /content/texts directory!")
            return

        print(f"Found {len(text_files)} text files to process.")

        # Determine the base message for animation
        if use_voice_clone and clone_audio_path:
            base_message = f"Generating audio with voice cloning for {len(text_files)} files..."
            clone_info = f"Using voice cloning with: {os.path.basename(clone_audio_path)}"
            transcript_preview = clone_transcript[:200] + "..." if len(clone_transcript) > 200 else clone_transcript
            print(clone_info)
            print(f"Transcript: {transcript_preview}")
        elif seed is not None:
            base_message = f"Generating audio with seed {seed} for {len(text_files)} files..."
            print(f"Using seed: {seed} for consistent voices")
        else:
            base_message = f"Generating audio for {len(text_files)} files..."
            print("Using random voice generation")

        # Start the loading animation
        start_loading_animation(output_area, base_message)

        # Process each file
        try:
            for i, file_path in enumerate(text_files):
                # Extract file name without extension
                file_name = os.path.basename(file_path)
                base_name = os.path.splitext(file_name)[0]

                # Update animation message with current file progress
                current_message = f"Processing file {i+1}/{len(text_files)}: {file_name}..."
                stop_loading_animation()  # Stop current animation
                start_loading_animation(output_area, current_message)  # Start with new message

                # Load text from file
                text = load_text_from_file(file_path)
                if not text:
                    continue

                # Generate output path
                output_path = os.path.join("/content/recordings", base_name)

                # Generate audio (this is where the actual generation happens)
                audio_path = generate_audio(
                    text=text,
                    output_path=output_path,
                    output_format=output_format,
                    max_tokens=max_tokens,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_p=top_p,
                    cfg_filter_top_k=cfg_filter_top_k,
                    speed_factor=speed_factor,
                    seed=seed,
                    compute_dtype=compute_dtype,
                    use_torch_compile=use_torch_compile,
                    use_voice_clone=use_voice_clone,
                    clone_audio_path=clone_audio_path,
                    clone_transcript=clone_transcript
                )

        finally:
            # Always stop animation when done (success or error)
            stop_loading_animation()

            with output_area:
                clear_output()
                print(f"✅ Batch processing complete!")
                print(f"All {len(text_files)} audio files have been saved to /content/recordings")

                # Show final results
                for i, file_path in enumerate(text_files):
                    file_name = os.path.basename(file_path)
                    base_name = os.path.splitext(file_name)[0]

                    # Determine the final file extension
                    final_extension = output_format
                    final_audio_path = os.path.join("/content/recordings", f"{base_name}.{final_extension}")

                    if os.path.exists(final_audio_path):
                        print(f"[{i+1}] {file_name} → {base_name}.{final_extension}")
                        # Display the audio player for each generated file
                        display(ipd.Audio(final_audio_path, autoplay=False))
                    else:
                        print(f"[{i+1}] {file_name} → ERROR: File not generated")

# Create a validator function for numeric input fields
def create_validator(min_val, max_val, default_val, step=1):
    def validate(change):
        widget = change.owner
        value = change.new
        try:
            # Convert to appropriate numeric type
            if isinstance(step, int):
                num_value = int(value)
            else:
                num_value = float(value)

            # Validate range
            if num_value < min_val:
                widget.value = str(min_val)
            elif num_value > max_val:
                widget.value = str(max_val)
        except (ValueError, TypeError):
            # Reset to default if input is invalid
            widget.value = str(default_val)
    return validate

# Silently check and install dependencies
def check_and_install_dependencies():
    try:
        import pydub
    except ImportError:
        os.system('pip install pydub > /dev/null 2>&1')

    # Check if ffmpeg is installed
    if os.system('which ffmpeg > /dev/null 2>&1') != 0:
        os.system('apt-get update > /dev/null 2>&1 && apt-get install -y ffmpeg > /dev/null 2>&1')

# Create the UI interface
def create_batch_dia_ui():
    # First, ensure the required directories exist
    ensure_directories()

    # Get list of text files and clone audio files
    text_files = get_text_files()
    clone_files = get_clone_audio_files()
    file_count = len(text_files)
    clone_count = len(clone_files)

    # Load existing clone configuration if it exists
    existing_config = load_clone_config()

    # Input directory info
    directory_info = widgets.HTML(
        value=f"<h3>Found {file_count} text files in /content/texts</h3>" +
              "<small>Text files should be named part_1.txt, part_2.txt, etc.</small>"
    )

    # Clone directory info
    clone_info = widgets.HTML(
        value=f"<h3>Found {clone_count} audio files in /content/clone</h3>" +
              "<small>Add MP3 or WAV files to /content/clone for voice cloning</small>"
    )

    # Configuration status info
    config_status = widgets.HTML()
    if existing_config:
        config_status.value = f"<p><small><b>🔧 Saved Configuration:</b> {existing_config['clone_file']} with {existing_config['transcript_source']} (saved {existing_config['timestamp']})</small></p>"
    else:
        config_status.value = "<p><small><b>🔧 No saved configuration found.</b> Voice settings will be saved when you process files.</small></p>"

    # Refresh button for text files
    refresh_button = widgets.Button(
        description='Refresh File Lists',
        button_style='info',
        tooltip='Refresh the list of text and clone files',
        icon='refresh'
    )

    def on_refresh_button_clicked(b):
        nonlocal text_files, clone_files
        text_files = get_text_files()
        clone_files = get_clone_audio_files()
        file_count = len(text_files)
        clone_count = len(clone_files)

        directory_info.value = f"<h3>Found {file_count} text files in /content/texts</h3>" + \
                              "<small>Text files should be named part_1.txt, part_2.txt, etc.</small>"

        clone_info.value = f"<h3>Found {clone_count} audio files in /content/clone</h3>" + \
                          "<small>Add MP3 or WAV files to /content/clone for voice cloning</small>"

        # Update clone dropdown options
        clone_dropdown.options = ['None'] + [os.path.basename(f) for f in clone_files]

    refresh_button.on_click(on_refresh_button_clicked)

    # Voice Generation Mode selection
    voice_mode_title = widgets.HTML(value="<h3>Voice Generation Mode</h3>")

    voice_mode_tabs = widgets.Tab(
        children=[widgets.VBox([]), widgets.VBox([])],
        titles=['Use Voice Cloning', 'Use Seed']
    )
    voice_mode_tabs.set_title(0, 'Use Voice Cloning')
    voice_mode_tabs.set_title(1, 'Use Seed')

    # SEED MODE WIDGETS
    # Seed input for consistent voice across files
    seed_input = widgets.Text(
        value='',
        description='Voice Seed:',
        placeholder='Enter a number for consistent voices',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )
    seed_help = widgets.HTML(value="<small>Setting a specific seed value ensures the same voices will be used across all files.<br>Leave empty for random voices.</small>")

    # VOICE CLONING WIDGETS
    # Clone file selection
    default_clone = 'None'
    if existing_config and existing_config['clone_file'] in [os.path.basename(f) for f in clone_files]:
        default_clone = existing_config['clone_file']

    clone_dropdown = widgets.Dropdown(
        options=['None'] + [os.path.basename(f) for f in clone_files],
        value=default_clone,
        description='Click Here To Select A Voice to Clone:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )

    clone_preview_button = widgets.Button(
        description='Preview Audio',
        button_style='info',
        tooltip='Play the selected clone audio',
        icon='volume-up',
        layout=widgets.Layout(width='auto')
    )

    # Transcript source selection
    default_transcript_source = 'Use text file'
    if existing_config:
        default_transcript_source = existing_config['transcript_source']

    transcript_source = widgets.RadioButtons(
        options=['Use text file', 'Enter manually'],
        value=default_transcript_source,
        description='Transcript source:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )

    transcript_file_info = widgets.HTML(
        value="<small>Add a .txt file with the same name as your audio file in the /content/clone folder.<br>" +
              "For example, if your audio is 'voice_sample.mp3', create 'voice_sample.txt' with the transcript.</small>"
    )

    # Clone transcript input
    default_transcript_text = example_transcript
    if existing_config and existing_config.get('transcript_text'):
        default_transcript_text = existing_config['transcript_text']

    clone_transcript = widgets.Textarea(
        value=default_transcript_text,
        placeholder='Enter the exact transcript of the reference audio',
        description='Transcript:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='100%', height='150px')
    )
    clone_transcript_help = widgets.HTML(
        value="<small>Enter the exact transcript of what is spoken in the reference audio.<br>" +
              "Use [S1], [S2] tags for different speakers, just like in your regular text files.<br>" +
              "Accuracy is important for good voice cloning results!</small>"
    )

    # Function to toggle transcript input visibility
    def toggle_transcript_input(change):
        if change.new == 'Use text file':
            clone_transcript.layout.display = 'none'
            clone_transcript_help.layout.display = 'none'
            transcript_file_info.layout.display = 'block'
        else:
            clone_transcript.layout.display = 'block'
            clone_transcript_help.layout.display = 'block'
            transcript_file_info.layout.display = 'none'

            # Check if there's a transcript file and load it as a starting point
            if clone_dropdown.value != 'None':
                for file_path in clone_files:
                    if os.path.basename(file_path) == clone_dropdown.value:
                        transcript = get_transcript_for_audio(file_path)
                        if transcript:
                            clone_transcript.value = transcript
                        break

    transcript_source.observe(toggle_transcript_input, names='value')

    # Function to check for transcript file when audio is selected
    def on_clone_dropdown_change(change):
        if change.new != 'None' and transcript_source.value == 'Use text file':
            # Check if there's a transcript file
            for file_path in clone_files:
                if os.path.basename(file_path) == change.new:
                    transcript = get_transcript_for_audio(file_path)
                    if transcript:
                        transcript_file_info.value = f"<small style='color:green'>✓ Found transcript file for {change.new}</small>"
                    else:
                        transcript_file_info.value = f"<small style='color:orange'>⚠ No transcript file found for {change.new}.<br>Create one at /content/clone/{os.path.splitext(change.new)[0]}.txt</small>"
                    break

    clone_dropdown.observe(on_clone_dropdown_change, names='value')

    # Clone audio preview area
    clone_preview_area = widgets.Output()

    def on_clone_preview_button_clicked(b):
        with clone_preview_area:
            clear_output()
            if clone_dropdown.value != 'None':
                # Find full path of selected clone file
                selected_file = None
                for file_path in clone_files:
                    if os.path.basename(file_path) == clone_dropdown.value:
                        selected_file = file_path
                        break

                if selected_file:
                    print(f"Playing: {clone_dropdown.value}")
                    display(ipd.Audio(selected_file, autoplay=True))
                else:
                    print("File not found")
            else:
                print("No clone file selected")

    clone_preview_button.on_click(on_clone_preview_button_clicked)

    # Arrange voice cloning widgets
    voice_clone_container = widgets.VBox([
        widgets.HBox([clone_dropdown, clone_preview_button]),
        clone_preview_area,
        transcript_source,
        transcript_file_info,
        clone_transcript,
        clone_transcript_help
    ])

    # Initial setup of transcript input visibility based on loaded config
    if default_transcript_source == 'Use text file':
        clone_transcript.layout.display = 'none'
        clone_transcript_help.layout.display = 'none'
    else:
        transcript_file_info.layout.display = 'none'

    # Arrange seed widgets
    seed_container = widgets.VBox([
        widgets.HBox([seed_input]),
        seed_help
    ])

    # Set the tab contents - Voice Cloning first
    voice_mode_tabs.children = [voice_clone_container, seed_container]

    # Parameter input fields with validation
    max_tokens_input = widgets.Text(
        value='3072',
        description='Max Tokens:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    max_tokens_input.observe(create_validator(860, 12000, 3072), names='value')
    max_tokens_help = widgets.HTML(value="<small>Range: 860-12000<br>Controls the maximum length of the generated audio (more tokens = longer audio).</small>")

    cfg_scale_input = widgets.Text(
        value='3.0',
        description='CFG Scale:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    cfg_scale_input.observe(create_validator(1.0, 15.0, 3.0, 0.1), names='value')
    cfg_scale_help = widgets.HTML(value="<small>Range: 1.0-15.0<br>Higher values increase adherence to the text prompt.</small>")

    temperature_input = widgets.Text(
        value='1.2',
        description='Temperature:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    temperature_input.observe(create_validator(0.5, 2.0, 1.2, 0.1), names='value')
    temperature_help = widgets.HTML(value="<small>Range: 0.5-2.0<br>Lower values make the output more deterministic, higher values increase randomness.</small>")

    top_p_input = widgets.Text(
        value='0.95',
        description='Top P:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    top_p_input.observe(create_validator(0.5, 1.0, 0.95, 0.01), names='value')
    top_p_help = widgets.HTML(value="<small>Range: 0.5-1.0<br>Filters vocabulary to the most likely tokens cumulatively reaching probability P.</small>")

    cfg_filter_top_k_input = widgets.Text(
        value='45',
        description='CFG Filter Top K:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    cfg_filter_top_k_input.observe(create_validator(15, 100, 45, 5), names='value')
    cfg_filter_top_k_help = widgets.HTML(value="<small>Range: 15-100<br>Top k filter for CFG guidance. Controls how many tokens are considered during generation.</small>")

    speed_factor_input = widgets.Text(
        value='0.92',
        description='Speed Factor:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    speed_factor_input.observe(create_validator(0.5, 1.5, 0.94, 0.01), names='value')
    speed_factor_help = widgets.HTML(value="<small>Range: 0.5-1.5<br>Adjusts the speed of the generated audio (1.0 = original speed).</small>")

    # Output format dropdown - default to WAV
    output_format_dropdown = widgets.Dropdown(
        options=['wav', 'mp3'],
        value='wav',  # Default to WAV
        description='Output Format:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    output_format_help = widgets.HTML(value="<small>File format for saving the audio files.</small>")

    compute_dtype_dropdown = widgets.Dropdown(
        options=['float16', 'float32', 'bfloat16'],
        value='float16',
        description='Compute Precision:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='50%')
    )
    compute_dtype_help = widgets.HTML(value="<small>Numerical precision for model calculations. Lower precision is faster but may reduce quality.</small>")

    use_torch_compile_checkbox = widgets.Checkbox(
        value=True,
        description='Use torch.compile (faster after first run)',
        style={'description_width': 'initial'}
    )
    torch_compile_help = widgets.HTML(value="<small>Optimizes PyTorch code for faster execution. First run is slower but subsequent runs are faster.</small>")

    # Output area
    output_area = widgets.Output()

    # Process button
    process_button = widgets.Button(
        description='Process All Files',
        button_style='primary',
        tooltip='Process all text files in sequence',
        icon='play'
    )

    # Preview selected file button
    preview_dropdown = widgets.Dropdown(
        options=['Select a file to preview...'] + [os.path.basename(f) for f in text_files],
        value='Select a file to preview...',
        description='Preview File:',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='70%')
    )

    preview_area = widgets.Output()

    def on_preview_change(change):
        with preview_area:
            clear_output()
            selected = change.new
            if selected != 'Select a file to preview...':
                # Find the full path
                for file_path in text_files:
                    if os.path.basename(file_path) == selected:
                        text = load_text_from_file(file_path)
                        print(f"Preview of {selected}:\n\n{text}")
                        break

    preview_dropdown.observe(on_preview_change, names='value')

    # Button click handler
    def on_process_button_clicked(b):
        # Get values from widgets and convert to appropriate types
        max_tokens = int(max_tokens_input.value)
        cfg_scale = float(cfg_scale_input.value)
        temperature = float(temperature_input.value)
        top_p = float(top_p_input.value)
        cfg_filter_top_k = int(cfg_filter_top_k_input.value)
        speed_factor = float(speed_factor_input.value)

        # Handle voice generation mode
        selected_tab = voice_mode_tabs.selected_index

        # Default: no voice cloning, no seed
        use_voice_clone = False
        clone_audio_path = None
        clone_transcript_text = None
        seed_value = None

        if selected_tab == 0:  # Voice cloning mode
            # Get clone file path
            if clone_dropdown.value != 'None':
                use_voice_clone = True

                # Find full path of selected clone file
                selected_clone_file = None
                for file_path in clone_files:
                    if os.path.basename(file_path) == clone_dropdown.value:
                        selected_clone_file = file_path
                        break

                clone_audio_path = selected_clone_file

                # Get transcript based on source selection
                if transcript_source.value == 'Use text file':
                    # Try to get transcript from file
                    transcript_from_file = get_transcript_for_audio(clone_audio_path)
                    if transcript_from_file:
                        clone_transcript_text = transcript_from_file
                    else:
                        print("No transcript file found. Please create one or switch to manual entry.")
                        use_voice_clone = False
                else:
                    # Use transcript from textarea
                    clone_transcript_text = clone_transcript.value

                # Save the clone configuration for future use
                if use_voice_clone:
                    save_clone_config(
                        clone_file=clone_dropdown.value,
                        transcript_source=transcript_source.value,
                        transcript_text=clone_transcript_text if transcript_source.value == 'Enter manually' else "",
                        cfg_scale=cfg_scale,
                        speed_factor=speed_factor
                    )
        else:  # Seed mode
            # Handle optional seed
            try:
                seed_value = int(seed_input.value) if seed_input.value else None
            except ValueError:
                print("Warning: Invalid seed value. Using random seed instead.")
                seed_value = None

        # Get output format
        output_format = output_format_dropdown.value

        # Start batch processing
        batch_process_files(
            output_format=output_format,
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            speed_factor=speed_factor,
            seed=seed_value,
            compute_dtype=compute_dtype_dropdown.value,
            use_torch_compile=use_torch_compile_checkbox.value,
            use_voice_clone=use_voice_clone,
            clone_audio_path=clone_audio_path,
            clone_transcript=clone_transcript_text,
            output_area=output_area
        )

    process_button.on_click(on_process_button_clicked)

    # Create accordion for advanced parameters - only show CFG and Speed Factor
    advanced_params = widgets.Accordion(
        children=[
            widgets.VBox([
                widgets.HBox([cfg_scale_input, cfg_scale_help]),
                widgets.HBox([speed_factor_input, speed_factor_help])
            ])
        ],
        selected_index=None
    )
    advanced_params.set_title(0, 'Advanced Parameters')

    # Layout the widgets
    ui = widgets.VBox([
        widgets.HTML(value="<h2>Dia-1.6B Batch Text-to-Speech Processor</h2>"),
        widgets.HTML(value="<b>This tool processes all text files in the /content/texts folder sequentially and saves audio to /content/recordings</b>"),
        config_status,
        widgets.HBox([directory_info]),
        widgets.HBox([refresh_button]),
        clone_info,
        widgets.HBox([preview_dropdown]),
        preview_area,
        voice_mode_title,
        voice_mode_tabs,
        widgets.HBox([max_tokens_input, max_tokens_help]),
        widgets.HBox([output_format_dropdown, output_format_help]),
        advanced_params,
        process_button,
        output_area
    ], layout=widgets.Layout(width='90%'))

    display(ui)

# Main execution - silently check dependencies and run
check_and_install_dependencies()
create_batch_dia_ui()

a context for the code:

