import sounddevice as sd
import numpy as np
import queue
import threading
import os
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from faster_whisper import WhisperModel
# Replace the existing moviepy import block with this
import sys
try:
    import moviepy
    from moviepy import VideoFileClip
    print(f"✅ MoviePy version {moviepy.__version__} loaded successfully")
    print(f"MoviePy module path: {moviepy.__file__}")
except ImportError as e:
    print(f"⚠️ Failed to import moviepy: {e}")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print("Please ensure moviepy is installed correctly using 'pip install moviepy'")
    print("Run 'pip show moviepy' to verify installation details")
    print("Check the moviepy installation directory: C:\\Users\\harsh\\OneDrive\\Desktop\\Summer Internship - Sujata Mam\\venv\\lib\\site-packages\\moviepy")
    exit(1)   
# ==== CONFIGURATION ====
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load Whisper model for audio transcription
print("🔁 Loading Whisper model...")
whisper_model = WhisperModel("base", compute_type="int8")
print("✅ Whisper model loaded!")

# Load Gemini model for Q&A generation
gemini_model = genai.GenerativeModel("models/gemini-1.5-flash")

# Generate timestamp-based filenames
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
transcript_file = f"transcripts/transcript_{timestamp}.txt"
summary_file = f"transcripts/qa_summary_{timestamp}.txt"

# ==== AUDIO SETTINGS ====
SAMPLING_RATE = 16000
CHUNK_DURATION = 3
CHUNK_SIZE = int(SAMPLING_RATE * CHUNK_DURATION)
MERGE_SECONDS = 9

# ==== INIT ====
audio_queue = queue.Queue()
audio_buffer = []

if not os.path.exists("transcripts"):
    os.makedirs("transcripts")

# ==== AUDIO CALLBACK ====
def audio_callback(indata, frames, time, status):
    if status:
        print("⚠️ Mic Error:", status)
    audio_queue.put(indata.copy())

# ==== SAVE TRANSCRIPT ====
def save_segment_to_file(text):
    with open(transcript_file, "a", encoding="utf-8") as f:
        f.write(text + "\n")

# ==== GEMINI Q&A ====
# Update the generate_qa_summary function
def generate_qa_summary(transcript_path):
    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = f.read()

    # Split transcript into chunks (approx 10,000 characters per chunk)
    max_chunk_size = 10000
    transcript_lines = transcript.splitlines()
    chunks = []
    current_chunk = []
    current_length = 0

    for line in transcript_lines:
        line_length = len(line) + 1  # +1 for newline
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
        print(f"Processing chunk {i}/{len(chunks)}...")
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
            summary = response.text
            all_summaries.append(f"--- Chunk {i} ---\n{summary}")
        except Exception as e:
            print(f"⚠️ Error processing chunk {i}: {e}")
            all_summaries.append(f"--- Chunk {i} ---\nError: Could not generate Q&A for this chunk")

    # Combine all summaries
    combined_summary = "\n\n".join(all_summaries)

    # Save to file
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(combined_summary)

    print("\n📄 Q&A Summary:\n")
    print(combined_summary)
    
# ==== TRANSCRIBE THREAD ====
def transcribe_audio():
    while True:
        audio_chunk = audio_queue.get()
        audio_buffer.append(audio_chunk)

        max_chunks = MERGE_SECONDS // CHUNK_DURATION
        if len(audio_buffer) > max_chunks:
            audio_buffer.pop(0)

        combined_audio = np.concatenate(audio_buffer, axis=0).flatten()

        print("🧠 Transcribing merged audio...")
        segments, _ = whisper_model.transcribe(combined_audio, language="en")

        print("\n🎯 --- Captions ---")
        for segment in segments:
            print(f"[{segment.start:.1f}s - {segment.end:.1f}s]: {segment.text}")
            save_segment_to_file(segment.text)
        print("-------------------\n")


# Update the process_video function
def process_video(video_path):
    print(f"🔄 Processing video: {video_path}")
    try:
        video = VideoFileClip(video_path)
        audio = video.audio
        audio_path = f"temp_audio_{timestamp}.wav"
        audio.write_audiofile(audio_path)  # Changed from write_wav to write_audiofile
        audio.close()
        video.close()
        
        # Transcribe the extracted audio
        print("🧠 Transcribing video audio...")
        segments, _ = whisper_model.transcribe(audio_path, language="en")
        
        print("\n🎯 --- Video Captions ---")
        for segment in segments:
            print(f"[{segment.start:.1f}s - {segment.end:.1f}s]: {segment.text}")
            save_segment_to_file(segment.text)
        print("-------------------\n")
        
        # Clean up temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)
        
        return transcript_file
    except Exception as e:
        print(f"⚠️ Error processing video: {e}")
        return None

# ==== MAIN ====
if __name__ == "__main__":
    try:
        input_type = input("🎙️ Enter 'live' for microphone or 'video' for video file: ").strip().lower()
        
        if input_type == "live":
            print("🎙️ Speak into your mic... (Press Ctrl+C to stop)\n")
            threading.Thread(target=transcribe_audio, daemon=True).start()
            with sd.InputStream(samplerate=SAMPLING_RATE, channels=1, callback=audio_callback, blocksize=CHUNK_SIZE):
                while True:
                    pass
        elif input_type == "video":
            video_path = input("📹 Enter video file path: ").strip()
            if not os.path.exists(video_path):
                print(f"⚠️ Video file {video_path} not found!")
            else:
                transcript_path = process_video(video_path)
                if transcript_path:
                    print(f"\n🛑 Video processing complete. Transcript saved to {transcript_file}")
                    generate_qa_summary(transcript_file)
                    print(f"\n✅ Q&A Summary saved to {summary_file}")
        else:
            print("⚠️ Invalid input! Use 'live' or 'video'.")
    except KeyboardInterrupt:
        if input_type == "live":
            print(f"\n🛑 Live captioning stopped. Transcript saved to {transcript_file}")
            generate_qa_summary(transcript_file)
            print(f"\n✅ Q&A Summary saved to {summary_file}")
