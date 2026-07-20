import cv2
import pytesseract
from dataclasses import dataclass
import csv

# Caminho do executável do Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@dataclass
class ClubStats:
    # Times
    home_team: str = ""
    away_team: str = ""

    # Placar
    home_score: int = 0
    away_score: int = 0

    # Estatísticas
    possession_home: int = 0
    possession_away: int = 0

    shots_home: int = 0
    shots_away: int = 0

    expected_goals_home: float = 0.0
    expected_goals_away: float = 0.0

    passes_home: int = 0
    passes_away: int = 0

    tackles_home: int = 0
    tackles_away: int = 0

    tackles_won_home: int = 0
    tackles_won_away: int = 0

    interceptions_home: int = 0
    interceptions_away: int = 0

    saves_home: int = 0
    saves_away: int = 0

    fouls_committed_home: int = 0
    fouls_committed_away: int = 0

    offsides_home: int = 0
    offsides_away: int = 0

    corners_home: int = 0
    corners_away: int = 0

    fouls_home: int = 0
    fouls_away: int = 0

    penalties_home: int = 0
    penalties_away: int = 0

    yellow_cards_home: int = 0
    yellow_cards_away: int = 0

    red_cards_home: int = 0
    red_cards_away: int = 0

    dribble_accuracy_home: int = 0
    dribble_accuracy_away: int = 0

    shot_accuracy_home: int = 0
    shot_accuracy_away: int = 0

    pass_accuracy_home: int = 0
    pass_accuracy_away: int = 0


# ======================
# Carrega a imagem
# ======================

img = cv2.imread("images/club-img.png")

if img is None:
    raise FileNotFoundError("Imagem não encontrada!")

texto = pytesseract.image_to_string(img, lang="eng")

print(texto)


# ======================
# Dados de teste
# ======================

stats = ClubStats(
    home_team="Crystal Palace",
    away_team="Tottenham",

    home_score=2,
    away_score=4,

    possession_home=62,
    possession_away=38,

    shots_home=16,
    shots_away=10,

    expected_goals_home=4.3,
    expected_goals_away=5.7,

    passes_home=291,
    passes_away=299,

    tackles_home=19,
    tackles_away=23,

    tackles_won_home=2,
    tackles_won_away=12,

    interceptions_home=28,
    interceptions_away=18,

    saves_home=2,
    saves_away=3,

    fouls_committed_home=2,
    fouls_committed_away=2,

    offsides_home=0,
    offsides_away=0,

    corners_home=2,
    corners_away=1,

    fouls_home=1,
    fouls_away=1,

    penalties_home=1,
    penalties_away=1,

    yellow_cards_home=1,
    yellow_cards_away=0,

    red_cards_home=1,
    red_cards_away=0,

    dribble_accuracy_home=91,
    dribble_accuracy_away=97,

    shot_accuracy_home=31,
    shot_accuracy_away=60,

    pass_accuracy_home=91,
    pass_accuracy_away=86
)


# ======================
# Salva no CSV
# ======================

with open("output/club_stats.csv", "a", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=ClubStats.__annotations__.keys()
    )

    if file.tell() == 0:
        writer.writeheader()

    writer.writerow(stats.__dict__)

print("Dados do clube salvos com sucesso!")