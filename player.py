import cv2
import pytesseract
from dataclasses import dataclass
import csv


@dataclass
class MatchStats:
    player_name: str
    player_rating: float
    goals: int
    assists: int
    shots: int
    passes: int
    dribbles: int
    tackles: int


# ==========================
# Verifica se a imagem existe
# ==========================
img = cv2.imread("images/player-img.png")

if img is None:
    raise FileNotFoundError("A imagem 'player-img.png' não foi encontrada.")

print(f"Imagem carregada com sucesso! Tamanho: {img.shape}")


# ==========================================
# Dados de teste (depois virão do OCR)
# ==========================================
stats = MatchStats(
    player_name="Nycolas",
    player_rating=6.6,
    goals=0,
    assists=0,
    shots=2,
    passes=5,
    dribbles=10,
    tackles=2
)


# ==========================
# Salva no CSV
# ==========================
with open("output/match_stats.csv", "a", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=MatchStats.__annotations__.keys()
    )

    # Escreve o cabeçalho apenas se o arquivo estiver vazio
    if file.tell() == 0:
        writer.writeheader()

    # Escreve os dados
    writer.writerow(stats.__dict__)

print("Dados salvos com sucesso!")
