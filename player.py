import cv2
import pytesseract
import csv
import os
import re

# Caminho do Tesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

os.makedirs("debug", exist_ok=True)
os.makedirs("output", exist_ok=True)

# Recorta uma região

def recortar(imagem, regiao):
    return imagem[
        regiao["y"]:regiao["y"] + regiao["h"],
        regiao["x"]:regiao["x"] + regiao["w"]
    ]

# Pré-processamento para OCR (Otsu = threshold automático)

def preparar_ocr(imagem, escala=4):

    if len(imagem.shape) == 3:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

    imagem = cv2.resize(
        imagem,
        None,
        fx=escala,
        fy=escala,
        interpolation=cv2.INTER_CUBIC
    )

    _, imagem = cv2.threshold(
        imagem,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return imagem

# OCR

def ocr_texto(imagem, config):

    texto = pytesseract.image_to_string(
        imagem,
        lang="eng",
        config=config
    )

    return texto.strip()

# Lista fixa das 17 estatísticas da tela "DESEMPENHO
# INDIVIDUAL", sempre na mesma ordem

ROTULOS_TABELA = [
    "gols",
    "assistencias",
    "finalizacoes",
    "precisao_finalizacoes_pct",
    "passes",
    "precisao_passes_pct",
    "dribles",
    "taxa_dribles_certos_pct",
    "divididas",
    "taxa_divididas_ganhas_pct",
    "impedimentos",
    "faltas_cometidas",
    "posses_bola_ganhas",
    "perdas_posse_bola",
    "minutos_jogados_media_time",
    "distancia_percorrida_media_time_km",
    "distancia_corrida_media_time_km",
]


def parse_tabela(texto):
    linhas = [l for l in texto.split("\n") if l.strip()]

    # Descarta linhas de "lixo" no topo que não tem letras
    # (ex: '§', 'oe', ':') — sobra só as 17 linhas de dado
    linhas = [l for l in linhas if re.search(r"[A-Za-zÀ-ÿ]{3,}", l)]

    jogador = {}
    time = {}

    for rotulo, linha in zip(ROTULOS_TABELA, linhas):

        # Pega todos os números da linha (aceita vírgula decimal)
        numeros = re.findall(r"\d+(?:[.,]\d+)?", linha)

        jogador[rotulo] = numeros[0] if len(numeros) >= 1 else ""
        time[rotulo] = numeros[1] if len(numeros) >= 2 else ""

    return jogador, time


# Leitura de um jogador

def ler_jogador(caminho_imagem):

    img = cv2.imread(caminho_imagem)

    if img is None:
        raise FileNotFoundError(f"Imagem '{caminho_imagem}' não encontrada!")

    altura, largura = img.shape[:2]

    print(f"\nLendo: {caminho_imagem}")
    print(f"Resolução: {largura}x{altura}")

    if (largura, altura) == (1360, 768):

        regioes = {
            "nome": {"x": 191, "y": 150, "w": 57, "h": 41},
            "overall": {"x": 66, "y": 146, "w": 26, "h": 30},
            "tabela": {"x": 884, "y": 158, "w": 428, "h": 536},
        }

    elif (largura, altura) == (1920, 1080):

        regioes = {
            "nome": {"x": 266, "y": 208, "w": 89, "h": 56},
            "overall": {"x": 93, "y": 209, "w": 40, "h": 36},
            "tabela": {"x": 1258, "y": 226, "w": 594, "h": 753},
        }

    else:
        raise ValueError("Resolução não suportada!")

    nome = recortar(img, regioes["nome"])
    overall = recortar(img, regioes["overall"])
    tabela = recortar(img, regioes["tabela"])

    nome_proc = preparar_ocr(nome, escala=4)
    overall_proc = preparar_ocr(overall, escala=4)
    tabela_proc = preparar_ocr(tabela, escala=2)

    nome_arquivo = os.path.splitext(os.path.basename(caminho_imagem))[0]

    cv2.imwrite(f"debug/{nome_arquivo}_nome.png", nome_proc)
    cv2.imwrite(f"debug/{nome_arquivo}_overall.png", overall_proc)
    cv2.imwrite(f"debug/{nome_arquivo}_tabela.png", tabela_proc)

    nome_texto = ocr_texto(
        nome_proc,
        "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    ).replace("\n", " ")

    overall_texto = ocr_texto(
        overall_proc,
        "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"
    ).replace("\n", "")

    tabela_texto = ocr_texto(
        tabela_proc,
        "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    )

    stats_jogador, stats_time = parse_tabela(tabela_texto)

    print("----------------------------")
    print("Nome OCR    :", nome_texto)
    print("Overall OCR :", overall_texto)
    print("Stats jogador:", stats_jogador)
    print("Stats time   :", stats_time)
    print("----------------------------")

    # Monta o registro final: nome, overall + uma coluna
    # por estatística (só do jogador, como você pediu)
    registro = {
        "nome": nome_texto,
        "overall": overall_texto,
    }
    registro.update(stats_jogador)

    return registro, stats_time


# Lista de imagens

imagens = [
    "images/player-img.png",
    "images/player2-img.png",
]

jogadores = []
times = []  # guardamos à parte, caso queira usar depois

for imagem in imagens:
    jogador, stats_time = ler_jogador(imagem)
    jogadores.append(jogador)
    times.append(stats_time)


# CSV 

campos = ["nome", "overall"] + ROTULOS_TABELA

with open(
    "output/jogadores.csv",
    "w",
    newline="",
    encoding="utf-8"
) as arquivo:

    writer = csv.DictWriter(arquivo, fieldnames=campos)
    writer.writeheader()
    writer.writerows(jogadores)

print("\nCSV criado com sucesso!")
print("\nJogadores encontrados:\n")

for jogador in jogadores:
    print(jogador["nome"], "-", jogador["overall"])
