import cv2
from ultralytics import YOLO
import torch
import time
import math
import numpy as np

# Classe para detecção de pose
class PoseDetector:
    def __init__(self):
        # Verifica se CUDA está disponível e usa GPU se possível
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8n-pose.pt').to(self.device)  # Carrega o modelo de pose
        self.cap = cv2.VideoCapture(0)  # Captura da webcam
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.tempo_cotovelo = {}
        self.tempo_inclinacao = {}
        self.tempo_inclinacao_frontal = {}
        self.tempoTorcao = {}
        self.tempoLombar = {}
        self.tempo_agachamento_incorreto = {}
        if not self.cap.isOpened():
            raise Exception("Erro ao acessar a webcam.")
        
    def detectar_pose(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Erro ao capturar o frame da webcam.")
            
            results = self.model(frame, device=self.device)
            annotated_frame = results[0].plot()
            mensagens = []  # Lista para armazenar mensagens de alertas

            for i, (box, detection) in enumerate(zip(results[0].boxes, results[0].keypoints)):
                cotoveloAcima = False
                inclinadoParaTras = False
                inclinadoFrontal = False
                torceu = False
                inclinaLombar = False
                agachamento_incorreto = False
                agachamento_iniciado = {}
            
                for pessoaKey in detection.xy:

                    # Verificar se o cotovelo está acima da cabeça
                    try:
                        cotoveloCheck = CotoveloAcimaCabeca(pessoaKey).calcular()
                        if cotoveloCheck:  # Se o cotovelo estiver acima da cabeça
                            cotoveloAcima = True
                    except Exception as e:
                        print(f"Erro ao calcular cotovelo acima da cabeça: {e}")
                        cotoveloAcima = False
                    
                    try:
                        inclinadaLombar = InclinacaoLombar(pessoaKey).calcular()
                        if inclinadaLombar:
                            inclinaLombar = True
                    except Exception as e:
                        print(f"Erro ao calcular Inclinacao para tras: {e}")
                        inclinaLombar = False
                        
                    # Verificar se a cabeça está abaixada com base no nariz e orelhas
                    try:
                        inclinacaoCheck = InclinacaoParaTras(pessoaKey).calcular()
                        if inclinacaoCheck:
                            inclinadoParaTras = True
                    except Exception as e:
                        print(f"Erro ao calcular Inclinacao para tras: {e}")
                        inclinadoParaTras = False

                    try:
                        inclinacaoFrontalCheck = inclinacaoFrontal(pessoaKey).calcular()
                        if inclinacaoFrontalCheck > 0.85:
                            inclinadoFrontal = True
                    except Exception as e:
                        print(f"Erro ao calcular inclinação frontal: {e}")
                        inclinadoFrontal = False
                        
                    try:
                        desalinhado = AlinhamentoOmbros(pessoaKey).verificar()
                        if desalinhado:
                            mensagens.append(f"Pessoa {i + 1} com ombros desalinhados!")
                            self.bordaVermelha(box, annotated_frame)
                    except Exception as e:
                        print(f"Erro ao calcular alinhamento de ombros: {e}")
                    try: 
                        agach = Agachamento(pessoaKey)
                        angulo_direito = agach.calcular_angulo_joelho("direito")
                        angulo_esquerdo = agach.calcular_angulo_joelho("esquerdo")
                        if angulo_direito < 115 and angulo_esquerdo < 155 and angulo_direito > 0 and angulo_esquerdo > 0:
                            agachamento_iniciado[i] = True
                        elif angulo_direito >= 115 and angulo_esquerdo >= 115 :
                            agachamento_iniciado[i] = False

                        if agachamento_iniciado.get(i, False) and (angulo_direito < 40 or angulo_direito > 80 or angulo_esquerdo < 40 or angulo_esquerdo > 80):
                            agachamento_incorreto = True
                    except Exception as e:
                        print("Erro ao calcular ângulo do joelho: {e}")
                        agachamento_incorreto = False

                    # Adicionar mensagens e as bordas coloridas
                    if cotoveloAcima:
                        if i not in self.tempo_cotovelo:
                            self.tempo_cotovelo[i] = time.time()
                        elif time.time() - self.tempo_cotovelo[i] >= 3:
                            self.bordaVermelha(box, annotated_frame)
                            mensagens.append(f"Pessoa {i + 1} levantou o braço por mais de 3 segundos!")
                    else:
                        if i in self.tempo_cotovelo:
                            del self.tempo_cotovelo[i]
                    
                    if inclinadoParaTras:
                        if i not in self.tempo_inclinacao:
                            self.tempo_inclinacao[i] = time.time()
                        elif time.time() - self.tempo_inclinacao[i] >= 3:
                            self.bordaVermelha(box, annotated_frame)
                            mensagens.append(f"Pessoa {i + 1} está inclinada para trás por mais de 3 segundos!")
                    else:
                        if i in self.tempo_inclinacao:
                            del self.tempo_inclinacao[i]
                    
                    if inclinadoFrontal:
                        if i not in self.tempo_inclinacao_frontal:
                            self.tempo_inclinacao_frontal[i] = time.time()
                        elif time.time() - self.tempo_inclinacao_frontal[i] >= 3:
                            self.bordaVermelha(box, annotated_frame)
                            mensagens.append(f"Pessoa {i + 1} está inclinada frontalmente por mais de 3 segundos!")
                    else:
                        if i in self.tempo_inclinacao_frontal:
                            del self.tempo_inclinacao_frontal[i]
                    
                    if inclinaLombar:
                        if i not in self.tempoLombar:
                            self.tempoLombar[i] = time.time()
                        elif time.time() - self.tempoLombar[i] >= 3:
                            self.bordaVermelha(box, annotated_frame)
                            mensagens.append(f"Pessoa {i + 1} está com inclinação lombar excessiva por mais de 3 segundos!")
                    else:
                        if i in self.tempoLombar:
                            del self.tempoLombar[i]

                    if agachamento_incorreto:
                        if i not in self.tempo_agachamento_incorreto:
                            self.tempo_agachamento_incorreto[i] = time.time()
                        elif time.time() - self.tempo_agachamento_incorreto[i] >= 3:
                            self.bordaVermelha(box, annotated_frame)
                            mensagens.append(f"Pessoa {i + 1} está agachando incorretamente por mais de 3 segundos")    
                    else:
                        if i in self.tempo_agachamento_incorreto:
                            del self.tempo_agachamento_incorreto[i]

                    try:
                        torceu = torcaoPescoco(pessoaKey).calcular()
                        if torceu:
                            if i not in self.tempoTorcao:
                                self.tempoTorcao[i] = time.time()
                            elif time.time() - self.tempoTorcao[i] >= 3:
                                self.bordaVermelha(box,annotated_frame)
                                mensagens.append(f"Alerta! Pessoa {i + 1} torceu o pescoço! por mais de 3 segundos!")
                            else:
                                if i in self.tempoTorcao:
                                    del self.tempoTorcao[i]
                    except:
                        print(f"Erro ao calcular torção: {e}")
                    


            return mensagens, annotated_frame

    def liberarRecursos(self):
        self.cap.release()
        cv2.destroyAllWindows()
        
    def bordaVermelha(self, box, annotated_frame):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)

# Classe para detectar cotovelo acima da cabeça
class CotoveloAcimaCabeca:
    def __init__(self, keypoints):
        self.cotoveloDireito = keypoints[8]  # Cotovelo direito (índice 8)
        self.cotoveloEsquerdo = keypoints[7]  # Cotovelo esquerdo (índice 7)
        self.nariz = keypoints[0]  # Nariz (índice 0)

    def calcular(self):
        cotoveloAcimaCabeca = False
        if self.cotoveloDireito[1] < self.nariz[1]:
            cotoveloAcimaCabeca = True
        elif self.cotoveloEsquerdo[1] < self.nariz[1]:
            cotoveloAcimaCabeca = True

        return cotoveloAcimaCabeca
    
class InclinacaoParaTras:
    def __init__(self, keypoints):
        # Pontos de referência para detectar a inclinação
        self.nariz = keypoints[0]  # Nariz (índice 0)
        self.ombroDireito = keypoints[2]  # Ombro direito (índice 2)
        self.ombroEsquerdo = keypoints[5]  # Ombro esquerdo (índice 5)
        self.quadrilDireito = keypoints[9]  # Quadril direito (índice 9)
        self.quadrilEsquerdo = keypoints[12]  # Quadril esquerdo (índice 12)
        self.cabeca = keypoints[1] if len(keypoints) > 1 else None  # Ponto da cabeça (opcional)

    def calcular(self):
        inclinacao_tras = False

        # Cálculo para verificar a inclinação
        ombro_medio_x = (self.ombroDireito[0] + self.ombroEsquerdo[0]) / 2
        quadril_medio_x = (self.quadrilDireito[0] + self.quadrilEsquerdo[0]) / 2
        nariz_x = self.nariz[0]

        # Verificação 1: Ombros devem estar significativamente atrás dos quadris
        ombros_inclinados = ombro_medio_x < quadril_medio_x - 50

        # Verificação 2: O nariz deve estar atrás dos quadris para garantir inclinação completa
        cabeca_inclinada = nariz_x < quadril_medio_x - 30

        # Se ambas as condições forem atendidas, consideramos que há inclinação para trás
        if ombros_inclinados and cabeca_inclinada:
            inclinacao_tras = True

        return inclinacao_tras
    
class inclinacaoFrontal:
    def __init__(self, keypoints):
        self.ombroDireito = keypoints[6]
        self.ombroEsquerdo = keypoints[5]
        self.quadrilDireito = keypoints[12]
        self.quadrilEsquerdo = keypoints[11]
        self.nariz = keypoints[0]
    
    def calcular(self):
        # Calcula o centro dos ombros e o centro dos quadris
        centroOmbrosY = (self.ombroDireito[1] + self.ombroEsquerdo[1]) / 2
        centroQuadrisY = (self.quadrilDireito[1] + self.quadrilEsquerdo[1]) / 2

        # Calcula a altura do corpo (da cabeça até o quadril)
        alturaCorpo = self.nariz[1] - centroQuadrisY

        # Calcula a inclinação proporcional entre os ombros e os quadris
        inclinacaoFrontal = (centroOmbrosY - centroQuadrisY) / alturaCorpo

        return inclinacaoFrontal
    
class AlinhamentoOmbros:
    def __init__(self, keypoints):
        self.ombro_direito = keypoints[6]
        self.ombro_esquerdo = keypoints[5]

    def verificar(self, limite_tolerancia=15):
        # Calcula a diferença na altura entre os ombros
        diferenca_altura = abs(self.ombro_direito[1] - self.ombro_esquerdo[1])
        return diferenca_altura > limite_tolerancia

class torcaoPescoco:
    def __init__(self, keypoints):
        self.nariz = keypoints[0]
        self.orelhaDi = keypoints[4] 
        self.orelhaEs = keypoints[3]   
        self.ombroEs = keypoints[5] 
        self.ombroDi = keypoints[6] 

    def calcular(self):  
        ombroDiExiste = True
        ombroEsExiste = True
        orelhaDiExiste = True
        orelhaEsExiste = True
        margem_percentual=15 # margem percentual para tolerância
        try:
            
            dOmbroDi_X = (self.nariz[0] - self.ombroDi[0]) ** 2
            dOmbroDi_Y = (self.nariz[1] - self.ombroDi[1]) ** 2
            distanciaOmbroDi = math.sqrt(dOmbroDi_X + dOmbroDi_Y)
        except:
            ombroDiExiste = False

        try:
            
            dOmbroEs_X = (self.nariz[0] - self.ombroEs[0]) ** 2
            dOmbroEs_Y = (self.nariz[1] - self.ombroEs[1]) ** 2
            distanciaOmbroEs = math.sqrt(dOmbroEs_X + dOmbroEs_Y)
        except:
            ombroEsExiste = False

        try:
            
            dOrelhaDi_X = (self.nariz[0] - self.orelhaDi[0]) ** 2
            dOrelhaDi_Y = (self.nariz[1] - self.orelhaDi[1]) ** 2
            distanciaOrelhaDi = math.sqrt(dOrelhaDi_X + dOrelhaDi_Y)
        except:
            orelhaDiExiste = False

        try:
            
            dOrelhaEs_X = (self.nariz[0] - self.orelhaEs[0]) ** 2
            dOrelhaEs_Y = (self.nariz[1] - self.orelhaEs[1]) ** 2
            distanciaOrelhaEs = math.sqrt(dOrelhaEs_X + dOrelhaEs_Y)
        except:
            orelhaEsExiste = False

        # Caso a pessoa esteja de frente para a câmera
        if ombroDiExiste and ombroEsExiste:
            margem = margem_percentual / 100
            diferenca_permitida = distanciaOmbroEs * margem

            # Verifica se a diferença entre as distâncias nariz-ombro está dentro da margem
            if abs(distanciaOmbroDi - distanciaOmbroEs) > diferenca_permitida:
                return True  
            else:
                return False  

       

        if ombroDiExiste and orelhaDiExiste and not orelhaEsExiste:
            # Se apenas ombro direito e orelha direita estão visíveis, provavelmente está virada para a esquerda
            if distanciaOrelhaDi > distanciaOmbroDi:
                return True  

        if ombroEsExiste and orelhaEsExiste and not orelhaDiExiste:
            # Se apenas ombro esquerdo e orelha esquerda estão visíveis, provavelmente está virada para a direita
            if distanciaOrelhaEs > distanciaOmbroEs:
                return True  
            
        return False  

class InclinacaoLombar:
    def __init__(self, keypoints):
        # Inicializa os pontos dos ombros e quadris
        self.ombro_esq = keypoints[5]  # Ombro esquerdo (índice 5)
        self.ombro_dir = keypoints[6]  # Ombro direito (índice 6)
        self.quadril_esq = keypoints[11]  # Quadril esquerdo (índice 11)
        self.quadril_dir = keypoints[12]  # Quadril direito (índice 12)

    def calcular(self):
        # Definição dos ângulos mínimos e máximos para risco de inclinação
        ANGULO_RISCO_MIN = 35
        ANGULO_RISCO_MAX = 75

        # Calcula a diferença de posição entre os ombros
        dy_ombro = self.ombro_dir[1] - self.ombro_esq[1]
        dx_ombro = self.ombro_dir[0] - self.ombro_esq[0]

        # Calcula a diferença de posição entre os quadris
        dy_quadril = self.quadril_dir[1] - self.quadril_esq[1]
        dx_quadril = self.quadril_dir[0] - self.quadril_esq[0]

        # Calcula o ângulo da inclinação da linha entre os ombros e os quadris
        angulo_ombro = abs(np.degrees(np.arctan2(dy_ombro, dx_ombro)))
        angulo_quadril = abs(np.degrees(np.arctan2(dy_quadril, dx_quadril)))

        # Verifica se os ângulos estão dentro da faixa de risco
        risco_ombro = ANGULO_RISCO_MIN <= angulo_ombro <= ANGULO_RISCO_MAX
        risco_quadril = ANGULO_RISCO_MIN <= angulo_quadril <= ANGULO_RISCO_MAX

        # Retorna True se qualquer uma das inclinações estiver na faixa de risco
        return risco_ombro or risco_quadril


class Agachamento:
    def __init__(self, keypoints):
        self.quadrilDireito = keypoints[12]
        self.quadrilEsquerdo = keypoints[11]
        self.joelhoDireito = keypoints[14]
        self.joelhoEsquerdo = keypoints[13]
       
    def calcular_angulo_joelho(self, lado):
        if lado == "direito":
            quadril, joelho = self.quadrilDireito, self.joelhoDireito
        else:
            quadril, joelho = self.quadrilEsquerdo, self.joelhoEsquerdo

        
        return abs(joelho[1] - quadril[1])