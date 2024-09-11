import torch
import yt_dlp
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

class Model:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Whisper Running on {self.device}')
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = "openai/whisper-large-v3"
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )
    
    def generate_text(self, audio_path):
        result = self.pipe(audio_path)
        return result['text']

def download_video(youtube_url, output_directory):
    out_name = f'{output_directory}output.%(ext)s'
    print(out_name)
    ydl_opts = {
        'format': 'bestaudio/best',  
        'outtmpl': f'{output_directory}\output.%(ext)s', 
        'postprocessors': [
            {'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'},  
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        yt = ydl.extract_info(youtube_url)
        filename = f"{output_directory}\output.mp3" 
        print('='*60)
        print(filename)
        print('='*60)
        ydl.download([youtube_url])
        return filename

def create_transcript_youtube(youtube_url):
    model = Model()

    output_directory = "downloads"
    file_name = download_video(youtube_url, output_directory)

    result = model.generate_text(file_name)
    print(result)
    return result

if __name__ == '__main__':

    result = create_transcript_youtube("https://www.youtube.com/watch?v=TBIjgBVFjVI")
    
    print("=" * 100)
    print(result)
    print("=" * 100)
