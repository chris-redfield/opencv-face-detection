# opencv-face-detection
simple opencv face detection experiment


## pré requisitos
Como a lista é grande, os pacotes estão disponíveis no arquivo requirements.txt e podem ser instalados com:

```console
pip install -r requirements.txt
```

Em geral os pacotes são: OpenCV, FastAI e seus pré-requisitos.

## como usar o programa
1. Na pasta raiz do projeto, treine a rede neural executando comando abaixo:

```console
python face-recognition-train.py
```
O modelo será treinado (baseado em uma resnet34) e salvo na pasta models em formato pickle (.pkl)

2. Inicie o programa rodando o script abaixo
```console
python face-detection.py
```
o programa procurará pela primeira câmera conectada ao computador e:
- localizará todos os rostos na imagem
- classificará o último rosto encontrado na imagem (por razão do tempo para o correto processamento e curadoria do dataset, optamos por classificar somente 1 rosto)

3. Para encerrar o programa, pressione a tecla 'q'