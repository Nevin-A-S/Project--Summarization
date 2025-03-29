import os
import torch
import yt_dlp
import logging
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from typing import Optional, Dict, Any

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('youtube_transcriber')

class WhisperModel:
    """A wrapper for Whisper speech-to-text model with efficient initialization and processing."""
    
    def __init__(self, model_id: str = "openai/whisper-large-v3", use_8bit: bool = False) -> None:
        """
        Initialize the Whisper model.
        
        Args:
            model_id: The Hugging Face model identifier
            use_8bit: Whether to use 8-bit quantization for lower memory usage
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f'Whisper running on {self.device}')
        
        # Determine precision based on hardware
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        # Load model with optimizations
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": True,
            "use_safetensors": True,
        }
        
        # Only use device_map="auto" when using 8-bit quantization or when GPU is available
        if use_8bit and self.device == 'cuda':
            model_kwargs["device_map"] = "auto"
            model_kwargs["quantization_config"] = {"load_in_8bit": True}
        else:
            # For regular loading, we'll move the model to device after loading
            pass
            
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            **model_kwargs
        )
        
        if "device_map" not in model_kwargs:
            self.model.to(self.device)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_id)
        
        # Create pipeline with batching and chunking support
        pipe_kwargs = {
            "model": self.model,
            "tokenizer": self.processor.tokenizer,
            "feature_extractor": self.processor.feature_extractor,
            "chunk_length_s": 30,  # Process 30-second chunks for memory efficiency
            "stride_length_s": 5,   # 5-second overlap between chunks
            "torch_dtype": self.torch_dtype,
        }
        
        # Only add device if not using device_map="auto"
        if "device_map" not in model_kwargs:
            pipe_kwargs["device"] = self.device
            
        self.pipe = pipeline(
            "automatic-speech-recognition",
            **pipe_kwargs
        )
    
    def generate_text(self, audio_path: str) -> str:
        """
        Transcribe audio file to text.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text
        """
        try:
            logger.info(f"Transcribing {audio_path}")
            result = self.pipe(
                audio_path, 
                return_timestamps=False,
                generate_kwargs={"task": "transcribe"}
            )
            return result['text']
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise

class YouTubeDownloader:
    """Handles YouTube video downloads with appropriate error handling."""
    
    def __init__(self, output_directory: str = "downloads") -> None:
        """
        Initialize the YouTube downloader.
        
        Args:
            output_directory: Directory to save downloaded files
        """
        self.output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)
    
    def download_audio(self, youtube_url: str) -> str:
        """
        Download audio from YouTube URL.
        
        Args:
            youtube_url: URL of the YouTube video
            
        Returns:
            Path to the downloaded audio file
        """
        # Generate a safe filename based on video ID
        video_id = youtube_url.split("watch?v=")[-1].split("&")[0]
        output_filename = f"{self.output_directory}/audio_{video_id}.mp3"
        
        # Check if file already exists to avoid redownloading
        if os.path.exists(output_filename):
            logger.info(f"Using existing download: {output_filename}")
            return output_filename
            
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{self.output_directory}/audio_{video_id}.%(ext)s',
            'postprocessors': [
                {'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'},
            ],
            'quiet': False,
            'no_warnings': False,
            'ignoreerrors': False,
        }
        
        try:
            logger.info(f"Downloading audio from {youtube_url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([youtube_url])
                logger.info(f"Downloaded to {output_filename}")
                return output_filename
        except Exception as e:
            logger.error(f"Download error: {str(e)}")
            raise

def create_transcript_youtube(youtube_url: str) -> str:
    """
    Create transcript from a YouTube video.
    
    Args:
        youtube_url: URL of the YouTube video
        
    Returns:
        Transcribed text from the video
    """
    try:
        # Initialize components
        output_directory = os.path.join(os.getcwd(), "downloads")
        downloader = YouTubeDownloader(output_directory)
        model = WhisperModel()
        
        # Download the audio
        audio_file = downloader.download_audio(youtube_url)
        
        # Transcribe the audio
        result = model.generate_text(audio_file)
        
        return result
    except Exception as e:
        logger.error(f"Transcription process failed: {str(e)}")
        raise

if __name__ == '__main__':
    # Example usage
    try:
        youtube_url = "https://www.youtube.com/watch?v=TBIjgBVFjVI"
        result = create_transcript_youtube(youtube_url)
        
        logger.info("=" * 80)
        logger.info("Transcription result:")
        logger.info(result)
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"Main process error: {str(e)}")