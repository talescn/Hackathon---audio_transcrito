import os
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
from dotenv import load_dotenv
import assemblyai as aai
import google.generativeai as genai
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time

# Carregar variáveis de ambiente
load_dotenv()

# Chaves das APIs
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def gravar_audio_dinamico(fs=16000, output="audio.wav", threshold=100, max_silence_duration=3):
    print("🎙️ Aguarde... escutando para iniciar gravação")

    buffer = []
    silence_start = None
    gravando = False
    bloco_tamanho = int(0.5 * fs)  # 0.5 segundo de áudio por bloco

    with sd.InputStream(samplerate=fs, channels=1) as stream:
        while True:
            bloco, _ = stream.read(bloco_tamanho)
            bloco = bloco.flatten()
            energia = np.sqrt(np.mean(bloco**2)) * 1000

            if energia > threshold:
                if not gravando:
                    print("🎤 Iniciando gravação...")
                    gravando = True
                buffer.append(bloco)
                silence_start = None  # resetar contador de silêncio
            else:
                if gravando:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start >= max_silence_duration:
                        print("🛑 Silêncio detectado. Encerrando gravação.")
                        break

    # Salvar áudio
    audio_final = np.concatenate(buffer)
    audio_final = np.int16(audio_final * 32767)
    write(output, fs, audio_final)
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

gravar_audio_dinamico()

