# opencv-face-detection

## pré requisitos

Em geral os pacotes são: OpenCV, FastAI e seus pré-requisitos, mas como a lista completa é grande, ela pode ser acessada no arquivo requirements.txt.

Para instalar todos os pacotes:

```console
pip install -r requirements.txt
```



## como usar o programa
1. Na pasta raiz do projeto, treine a rede neural executando comando abaixo:

```console
python face-recognition-train.py
```
O modelo será treinado e salvo na pasta models em formato pickle (.pkl). O dataset de treino é pequeno e o modelo do fastai ja é pré-treinado, portanto não é necessário o uso de GPU para o treinamento.

2. Inicie o programa rodando o script abaixo
```console
python face-detection.py
```
o programa procurará pela primeira câmera conectada ao computador e:
- localizará todos os rostos na imagem;
- classificará o último rosto encontrado na imagem em tempo real (por razão do tempo para o correto processamento e curadoria do dataset, optamos por classificar somente 1 rosto).

3. Para encerrar o programa, pressione a tecla 'q'


## Funcionamento

1. Detecção de rostos:
Para detecção de rostos utilizamos o método de detecção de objetos "Haar feature-based cascade classifiers", proposto por Paul Viola e Michael Jones no paper "Rapid Object Detection using a Boosted Cascade of Simple Features", de 2001. A implementação do opencv foi utilizada, sua documentação está disponível em https://docs.opencv.org/4.5.2/db/d28/tutorial_cascade_classifier.html .

2. Reconhecimento facial:
Para o reconhecimento facial, acabamos utilizando um classificador binário, em função do tempo de coleta do dataset e do prazo do projeto. Aqui foi utilizada uma rede neural convolucional residual (resnet) de 34 camadas, disponível pré treinada no FastAI. A rede foi retreinada no nosso dataset de 20 imagens, e salva para 'produção' em seguida.

## Limitações

Em função do prazo de entrega, o modelo foi treinado com imagens inteiras, portanto ele só classifica um rosto na imagem. Com mais investimento de tempo no projeto, é possível treinar e testar o modelo somente com os rostos 'recortados' que foram localizados pelo Haar Cascade. Dessa forma, o modelo estaria apto a classificar todos os rostos ao mesmo tempo, em tempo real.

## Resultados

![First test](https://raw.githubusercontent.com/chris-redfield/opencv-face-detection/main/img/test.png)
