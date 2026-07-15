import cv2

img = cv2.imread("images/player2-img.png")

if img is None:
    raise FileNotFoundError("Imagem não encontrada!")

# Descobre a resolução
altura, largura = img.shape[:2]

# Escala de visualização
escala = 0.75

preview = cv2.resize(
    img,
    None,
    fx=escala,
    fy=escala,
    interpolation=cv2.INTER_AREA
)

# Seleciona na imagem reduzida
x, y, w, h = cv2.selectROI("Imagem", preview)

# Converte para as coordenadas reais
x = int(x / escala)
y = int(y / escala)
w = int(w / escala)
h = int(h / escala)

print(f"""
# Região

x = {x}
y = {y}
w = {w}
h = {h}
""")

cv2.destroyAllWindows()