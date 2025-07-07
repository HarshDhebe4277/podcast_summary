import os
import sys
import time
import queue
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from faster_whisper import WhisperModel

# === Load MoviePy (Video support) ===
try:
    import moviepy
    from moviepy import VideoFileClip
    print(f"✅ MoviePy version {moviepy.__version__} loaded successfully")
    print(f"MoviePy module path: {moviepy.__file__}")
except ImportError as e:
    print(f"⚠️ Failed to import moviepy: {e}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print("Please install moviepy using 'pip install moviepy'")
    exit(1)

# === CONFIGURATION ===
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# === WHISPER MODEL ===
print("🔁 Loading Whisper model...")
whisper_model = WhisperModel("base", compute_type="int8")
print("✅ Whisper model loaded!")

# === GEMINI MODEL ===
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# === TIMESTAMPS & PATHS ===
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
TRANSCRIPT_DIR = "transcripts"
AUDIO_DIR = "audio"
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

transcript_file = os.path.join(TRANSCRIPT_DIR, f"transcript_{timestamp}.txt")
summary_file = os.path.join(TRANSCRIPT_DIR, f"qa_summary_{timestamp}.txt")
audio_file = os.path.join(AUDIO_DIR, f"audio_{timestamp}.wav")

# === AUDIO SETTINGS ===
SAMPLE_RATE = 16000
BLOCK_DURATION = 5  # seconds
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION)

# === INIT QUEUES ===
audio_queue = queue.Queue()
audio_frames = []

# === SAVE CAPTION TO FILE ===
def save_caption(text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {text}"
    print(f"\r📝 {line}", flush=True)
    with open(transcript_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# === AUDIO CALLBACK ===
def audio_callback(indata, frames, time_info, status):
    if status:
        print("⚠️ Mic Error:", status)
    audio_queue.put(indata.copy())
    audio_frames.append(indata.copy())

# === TRANSCRIPTION THREAD ===
def transcribe_worker():
    while True:
        block = audio_queue.get()
        if block is None:
            break
        block = block.flatten()
        segments, _ = whisper_model.transcribe(block, language="en", beam_size=1, vad_filter=True)
        for segment in segments:
            save_caption(segment.text.strip())

# === GEMINI Q&A ===
def generate_qa_summary(transcript_path):
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    max_chunk_size = 10000
    lines = transcript.splitlines()
    chunks, current_chunk, current_length = [], [], 0

    for line in lines:
        line_length = len(line) + 1
        if current_length + line_length > max_chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_length = line_length
        else:
            current_chunk.append(line)
            current_length += line_length
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    print(f"🧠 Generating Q&A summary with Gemini Flash ({len(chunks)} chunks)...")
    all_summaries = []

    for i, chunk in enumerate(chunks, 1):
        print(f"📄 Processing chunk {i}/{len(chunks)}...")
        prompt = f"""
         You are a helpful assistant. You are given a transcript chunk from a meeting or podcast.

        Your job is to extract meaningful and natural **Question & Answer pairs**.

        ⚠️ The transcript may contain **repetitive or overlapping lines** due to real-time merging — ignore duplicate or repeated content.

        🔍 A question does **not need to have a question mark**. Use your judgment:
        - If a sentence implies a request, clarification, or inquiry — treat it as a question.
        - You must analyze the meaning, not just punctuation.

        ✅ Format the output as:
        Q: [Extracted question]
        A: [Accurate and complete answer from transcript]

        🧠 Focus on:
        - Logically valid Q&A only
        - No duplication
        - Skipping filler, greetings, repeated lines

        Here is the transcript chunk:
        {chunk}
        """
        try:
            response = gemini_model.generate_content(prompt)
            summary = response.text if hasattr(response, "text") else "[Empty response]"
            all_summaries.append(f"--- Chunk {i} ---\n{summary}")
        except Exception as e:
            print(f"❌ Error with chunk {i}: {e}")
            all_summaries.append(f"--- Chunk {i} ---\nError: Failed to generate summary")

    final_output = "\n\n".join(all_summaries)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(final_output)

    print(f"\n✅ Q&A Summary saved to {summary_file}")
    print(final_output)

# === VIDEO TRANSCRIPTION ===
def process_video(video_path):
    print(f"🔄 Processing video: {video_path}")
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio_path = f"temp_audio_{timestamp}.wav"
        audio.write_audiofile(audio_path)
        audio.close()
        video.close()

        print("🧠 Transcribing video audio...")
        segments, _ = whisper_model.transcribe(audio_path, language="en")

        for segment in segments:
            print(f"[{segment.start:.1f}s - {segment.end:.1f}s]: {segment.text}")
            save_caption(segment.text)

        if os.path.exists(audio_path):
            os.remove(audio_path)

        return transcript_file
    except Exception as e:
        print(f"⚠️ Error processing video: {e}")
        return None

# === MAIN ===
if __name__ == "__main__":
    try:
        input_type = input("🎙️ Enter 'live' for microphone or 'video' for video file: ").strip().lower()

        if input_type == "live":
            print("🎙️ Listening... (Press Ctrl+C to stop)\n")
            threading.Thread(target=transcribe_worker, daemon=True).start()
            with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=BLOCK_SIZE):
                while True:
                    time.sleep(0.1)

        elif input_type == "video":
            video_path = input("📹 Enter video file path: ").strip()
            if not os.path.exists(video_path):
                print(f"⚠️ Video file {video_path} not found!")
            else:
                transcript_path = process_video(video_path)
                if transcript_path:
                    print(f"\n🛑 Video processing complete. Transcript saved to {transcript_path}")
                    generate_qa_summary(transcript_path)

        else:
            print("⚠️ Invalid input! Use 'live' or 'video'.")

    except KeyboardInterrupt:
        if input_type == "live":
            print(f"\n🛑 Stopping live transcription...")
            audio_queue.put(None)

            audio_np = np.concatenate(audio_frames, axis=0)
            sf.write(audio_file, audio_np, SAMPLE_RATE)
            print(f"💾 Audio saved to {audio_file}")
            print(f"📝 Transcript saved to {transcript_file}")

            generate_qa_summary(transcript_file)
