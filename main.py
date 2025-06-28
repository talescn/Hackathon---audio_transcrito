import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
import assemblyai as aai
import google.generativeai as genai

# Carregar variáveis de ambiente
load_dotenv()

# Chaves das APIs
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Grava áudio e salva como .wav
def gravar_audio(duration=5, fs=16000, output="audio.wav"):
    print("🎙️ Gravando... fale agora")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    audio = np.int16(audio * 32767)
    write(output, fs, audio)
    print(f"✅ Áudio gravado como {output}")

# Transcreve o áudio usando a AssemblyAI com idioma português
def transcrever_audio_assemblyai(caminho_audio="audio.wav"):
    print("⏳ Transcrevendo com AssemblyAI...")
    config = aai.TranscriptionConfig(
        language_code="pt",
        speech_model=aai.SpeechModel.best
    )
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(caminho_audio)

    if transcript.status == "error":
        raise RuntimeError(f"❌ Erro na transcrição: {transcript.error}")

    texto = transcript.text
    print("📝 Texto transcrito:", texto)
    
    return texto

# Usa o Gemini para gerar sugestão com base no texto transcrito
def obter_sugestao_com_gemini(texto_transcrito):
    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"Eu acabei de dizer: '{texto_transcrito}'. Sugira de forma curta como posso responder ou continuar falando com o cliente."

    resposta = model.generate_content(prompt)
    sugestao = resposta.text.strip()
    print("💡 Sugestão de continuação:", sugestao)
    return sugestao

# Fluxo principal
if __name__ == "__main__":
    gravar_audio(duration=5)
    texto = transcrever_audio_assemblyai("audio.wav")
    obter_sugestao_com_gemini(texto)
