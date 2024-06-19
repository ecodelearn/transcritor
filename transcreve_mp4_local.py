import subprocess
import whisper
import os
import sys
import torch
from deep_translator import GoogleTranslator
from langdetect import detect
from tqdm import tqdm
import numpy as np

# Função para dividir o texto em partes menores
def split_text(text, max_length):
    words = text.split(' ')
    current_length = 0
    current_part = []
    parts = []

    for word in words:
        if current_length + len(word) + 1 > max_length:
            parts.append(' '.join(current_part))
            current_part = [word]
            current_length = len(word) + 1
        else:
            current_part.append(word)
            current_length += len(word) + 1

    if current_part:
        parts.append(' '.join(current_part))

    return parts

# Verifica se o diretório foi fornecido como argumento
if len(sys.argv) < 2:
    print("Por favor, forneça o caminho para o diretório que contém os arquivos .mp4.")
    sys.exit(1)

input_directory = sys.argv[1]

# Solicita ao usuário o modelo Whisper a ser usado
model_choice = input("Escolha o modelo Whisper ('medium' ou 'large'): ").strip().lower()
if model_choice not in ['medium', 'large']:
    print("Modelo inválido. Escolha 'medium' ou 'large'.")
    sys.exit(1)

# Obtém a lista de arquivos .mp4 no diretório
mp4_files = [f for f in os.listdir(input_directory) if f.endswith('.mp4')]

if not mp4_files:
    print("Nenhum arquivo .mp4 encontrado no diretório fornecido.")
    sys.exit(1)

# Cria o diretório "textos_transcritos" se não existir
output_dir = "textos_transcritos"
os.makedirs(output_dir, exist_ok=True)

# Carrega o modelo Whisper com suporte a GPU
model = whisper.load_model(model_choice)

for mp4_file in mp4_files:
    mp4_file_path = os.path.join(input_directory, mp4_file)

    # Extrai o áudio do arquivo MP4
    mp3_file_path = mp4_file_path.replace('.mp4', '.mp3')
    command = ['ffmpeg', '-i', mp4_file_path, '-q:a', '0', '-map', 'a', mp3_file_path]
    subprocess.run(command, check=True)

    # Carrega o áudio
    audio = whisper.load_audio(mp3_file_path)
    sample_rate = whisper.audio.SAMPLE_RATE
    duration = len(audio) / sample_rate

    # Transcreve o arquivo de áudio em segmentos
    segments = np.array_split(audio, int(duration // 30) + 1)

    all_text = ""

    print(f"Transcrevendo áudio de {mp4_file}...")
    for segment in tqdm(segments, desc="Transcrição", unit="segmento"):
        segment_audio = whisper.pad_or_trim(segment)
        mel = whisper.log_mel_spectrogram(segment_audio).to(model.device)
        options = whisper.DecodingOptions(language="en")
        result = whisper.decode(model, mel, options)
        all_text += result.text + " "

    original_text = all_text.strip()
    print("Transcrição original salva.")

    # Detecta o idioma da transcrição
    detected_language = detect(original_text)
    print("Fazendo tradução...")

    # Define a língua de destino para a tradução
    target_language = 'pt' if detected_language == 'en' else 'en'

    # Dividir o texto em partes menores para tradução
    text_parts = split_text(original_text, 4999)
    translated_text = ""

    # Tradução para o idioma de destino
    translator = GoogleTranslator(source=detected_language, target=target_language)
    for part in text_parts:
        translated_text += translator.translate(part) + " "

    translated_text = translated_text.strip()

    # Salva a transcrição em um arquivo com o nome do arquivo original
    base_filename = os.path.splitext(os.path.basename(mp4_file_path))[0]
    transcribed_filename = os.path.join(output_dir, f"{base_filename}.txt")
    with open(transcribed_filename, 'w', encoding='utf-8') as f:
        f.write(f"Transcrição original:\n{original_text}\n\n")
        f.write(f"Transcrição traduzida para {target_language}:\n{translated_text}\n")
    print(f"Trabalho completo. Transcrição de {mp4_file} salva em: {os.path.abspath(transcribed_filename)}")

# Libera a memória da GPU
del model
torch.cuda.empty_cache()
