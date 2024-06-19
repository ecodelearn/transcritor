#!/bin/bash

# Solicita o link do vídeo ao usuário
read -p "Cole o link do vídeo: " video_url

# Executa o comando yt-dlp com o link fornecido
yt-dlp -x -f bestaudio --audio-format mp3 -o "%(id)s.%(ext)s" "$video_url"
