import cv2
import pytesseract
import csv
import os
import re
import glob
from pytesseract import Output

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

os.makedirs("debug", exist_ok=True)
os.makedirs("output", exist_ok=True)


def recortar(imagem, x, y, w, h):
    return imagem[y:y + h, x:x + w]


def preparar_ocr(imagem, escala=4):
    if len(imagem.shape) == 3:
        imagem = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    imagem = cv2.resize(imagem, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
    _, imagem = cv2.threshold(imagem, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return imagem


def ocr_texto(imagem, config):
    return pytesseract.image_to_string(imagem, lang="por+eng", config=config).strip()


def detectar_aba(caminho):
    nome = os.path.basename(caminho).lower()
    if "resumo" in nome:
        return "resumo"
    if "posse" in nome:
        return "posse"
    if "final" in nome:
        return "finalizacoes"
    if "passe" in nome:
        return "passes"
    if "defesa" in nome:
        return "defesa"
    if "gl" in nome:
        return "gl"
    return "resumo"


# Rótulos de cada aba, na ordem em que aparecem na tela.

ROTULOS_RESUMO = [
    "gols", "assistencias", "finalizacoes", "precisao_finalizacoes_pct",
    "passes", "precisao_passes_pct", "dribles", "taxa_dribles_certos_pct",
    "divididas", "taxa_divididas_ganhas_pct", "impedimentos", "faltas_cometidas",
    "posses_bola_ganhas", "perdas_posse_bola", "minutos_jogados_media_time",
    "distancia_percorrida_media_time_km", "distancia_corrida_media_time_km",
]

ROTULOS_POSSE = [
    None, "posse_de_bola_pct", "dribles", "dribles_completos", "taxa_dribles_certos_pct",
    "distancia_conduzindo_km", "faltas_sofridas", "penaltis_sofridos",
    "conducao_normal_pct", "conducao_protecao_pct", "conducao_lateral_pct",
    None, "superando_oponentes", "superando_oponentes_fintas", "no_meio_das_pernas", "adiantadas",
]

ROTULOS_FINALIZACOES = [
    None, "gols", "gols_esperados", "finalizacoes", "certas", "erradas", "bloqueadas", "precisao_pct",
    None, "normal", "colocada", "cabeceio", "rasteira", "de_voleio", "por_cobertura", "bola_parada",
]

ROTULOS_PASSES = [
    None, "assistencias", "assistencias_esperadas", "passes", "concluidos", "interceptados",
    "precisao_passe_pct", "para_impedimento",
    None, "rasteiro", "por_cobertura", "enfiada", "enfiada_por_cima", "cruzamento", "bola_parada",
]

ROTULOS_DEFESA = [
    None, "divididas_em_pe", "divididas_em_pe_ganhas", "taxa_divididas_em_pe_ganhas_pct",
    "carrinhos", "carrinhos_certos", "taxa_carrinhos_certos_pct", "interceptacoes",
    "bloqueios", "chutoes", "disputas_aereas_ganhas", "duelos_perdidos",
    None, "faltas_cometidas", "penaltis_cometidos", "gols_contra",
]

ROTULOS_GL = [
    None, "finalizacoes_contra", "finalizacoes_certas", "defesas", "gols_sofridos",
    "taxa_defesas_pct", "defesas_penaltis", "gols_sofridos_penaltis",
    None, "normal", "colocada", "cabeceio", "rasteira", "de_voleio", "por_cobertura", "bola_parada",
]

ABAS = {
    "resumo": ROTULOS_RESUMO,
    "posse": ROTULOS_POSSE,
    "finalizacoes": ROTULOS_FINALIZACOES,
    "passes": ROTULOS_PASSES,
    "defesa": ROTULOS_DEFESA,
    "gl": ROTULOS_GL,
}

# Coordenadas da tabela de estatísticas por resolução
COORDENADAS_TABELA = {
    (1920, 1080): {"x": 1258, "y": 226, "w": 594, "h": 753},
    (1360, 768): {"x": 884, "y": 158, "w": 428, "h": 536},
}


# Acha a posição Y de cada linha da tabela usando o OCR da coluna de RÓTULOS 
# Fazendo o tratamento do erro para caso aja um 0 em alguma das tabelas

def linhas_da_tabela(gray, x, y, w_label, h):
    crop = gray[y:y + h, x:x + w_label]
    big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dados = pytesseract.image_to_data(
        otsu, lang="por+eng", config="--oem 3 --psm 6", output_type=Output.DICT
    )

    linhas = {}
    for i in range(len(dados["text"])):
        texto = dados["text"][i].strip()
        if not texto:
            continue
        chave = (dados["block_num"][i], dados["par_num"][i], dados["line_num"][i])
        topo, altura = dados["top"][i], dados["height"][i]
        if chave not in linhas:
            linhas[chave] = {"top": topo, "bottom": topo + altura}
        linhas[chave]["bottom"] = max(linhas[chave]["bottom"], topo + altura)
        linhas[chave]["top"] = min(linhas[chave]["top"], topo)
        linhas[chave].setdefault("texto", []).append(texto)

    # descarta linhas sem nenhuma letra (ruído/pontuação solta que
    # o OCR às vezes "inventa" no topo da tabela e desalinha tudo)
    validas = [l for l in linhas.values() if re.search(r"[A-Za-zÀ-ÿ]", " ".join(l["texto"]))]

    ordenadas = sorted(validas, key=lambda l: l["top"])
    return [(y + l["top"] // 2, y + l["bottom"] // 2) for l in ordenadas]


def ocr_numero(gray, y_topo, y_fundo, x_ini, x_fim, pad=4):
    crop = gray[max(0, y_topo - pad):y_fundo + pad, x_ini:x_fim]
    big = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    _, otsu = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    texto = pytesseract.image_to_string(
        otsu, lang="por+eng", config="--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789,."
    ).strip().strip(".")
    return texto if texto else "0"


def ler_aba(caminho_imagem, aba):

    img = cv2.imread(caminho_imagem)
    if img is None:
        raise FileNotFoundError(f"Imagem '{caminho_imagem}' não encontrada!")

    altura_img, largura_img = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if (largura_img, altura_img) == (1360, 768):
        nome_reg = {"x": 191, "y": 148, "w": 100, "h": 45}
        overall_reg = {"x": 66, "y": 146, "w": 26, "h": 30}
    elif (largura_img, altura_img) == (1920, 1080):
        nome_reg = {"x": 266, "y": 206, "w": 150, "h": 60}
        overall_reg = {"x": 93, "y": 209, "w": 40, "h": 36}
    else:
        raise ValueError("Resolução não suportada!")

    tab = COORDENADAS_TABELA[(largura_img, altura_img)]

    # nome e overall aparecem no painel esquerdo em qualquer aba
    nome_img = preparar_ocr(recortar(img, **nome_reg))
    overall_img = preparar_ocr(recortar(img, **overall_reg))

    nome_texto = ocr_texto(
        nome_img, "--oem 3 --psm 6 -c preserve_interword_spaces=1"
    ).replace("\n", " ")

    overall_texto = ocr_texto(
        overall_img, "--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789"
    ).replace("\n", "")

    nome_arquivo = os.path.splitext(os.path.basename(caminho_imagem))[0]
    cv2.imwrite(f"debug/{nome_arquivo}_nome.png", nome_img)
    cv2.imwrite(f"debug/{nome_arquivo}_overall.png", overall_img)

    resultado = {"nome": nome_texto, "overall": overall_texto}

    rotulos = ABAS[aba]
    w_label = tab["w"] - 115
    linhas = linhas_da_tabela(gray, tab["x"], tab["y"], w_label, tab["h"])

    if aba == "resumo":
        x_jogador = (tab["x"] + tab["w"] - 160, tab["x"] + tab["w"] - 67)
        x_time = (tab["x"] + tab["w"] - 67, tab["x"] + tab["w"] + 4)

        for rotulo, (yt, yb) in zip(rotulos, linhas):
            resultado[rotulo] = ocr_numero(gray, yt, yb, *x_jogador)
            resultado[f"time_{rotulo}"] = ocr_numero(gray, yt, yb, *x_time)

    else:
        x_num = (tab["x"] + tab["w"] - 115, tab["x"] + tab["w"] + 4)

        for rotulo, (yt, yb) in zip(rotulos, linhas):
            if rotulo is None:
                continue
            resultado[f"{aba}_{rotulo}"] = ocr_numero(gray, yt, yb, *x_num)

    print(f"{nome_arquivo} [{aba}] -> nome={nome_texto!r} overall={overall_texto!r}")

    return resultado



PASTA_PLAYERS = os.path.join("images", "Players")

imagens = sorted(
    f for ext in ("*.png", "*.jpg", "*.jpeg")
    for f in glob.glob(os.path.join(PASTA_PLAYERS, ext))
)

print(f"\n{len(imagens)} imagem(ns) encontrada(s)")

jogadores = []
todos_campos = ["nome", "overall"]

registro = {}

for imagem in imagens:
    aba = detectar_aba(imagem)

    # Lê a imagem da aba correspondente
    dados = ler_aba(imagem, aba)

    if aba == "resumo" and registro:
        jogadores.append(registro)
        for campo in registro:
            if campo not in todos_campos:
                todos_campos.append(campo)
        registro = {}

    registro.update(dados)

# Salva o último jogador
if registro:
    jogadores.append(registro)
    for campo in registro:
        if campo not in todos_campos:
            todos_campos.append(campo)


with open("output/jogadores.csv", "w", newline="", encoding="utf-8") as arquivo:
    writer = csv.DictWriter(arquivo, fieldnames=todos_campos)
    writer.writeheader()
    writer.writerows(jogadores)

print("\nCSV criado com sucesso!")
