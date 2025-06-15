import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import threading
import cv2
from detector import PoseDetector  # Importando o detector de poses
import time

# Classe para a interface gráfica com Tkinter
class PoseDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Detecção de Pose com YOLOv8")
        self.root.geometry("1000x600")
        self.root.configure(bg="#2e3b4e")  # Cor de fundo moderna (esquema escuro)
        
        # Criando a interface gráfica
        self.setup_widgets()

    def setup_widgets(self):
        # Tornar a interface responsiva
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Adicionar o botão "Iniciar"
        self.start_button = tk.Button(self.root, text="Iniciar Detecção de Pose", command=self.run_detection, font=("Helvetica", 14), relief="raised", width=20, bg="#4CAF50", fg="white", activebackground="#45a049", activeforeground="white")
        self.start_button.grid(row=0, column=0, padx=20, pady=20)

        # Criar o rótulo para exibir a animação de carregamento
        self.loading_label = tk.Label(self.root, text="", font=("Helvetica", 50, "bold"), fg="#FF5733", bg="#2e3b4e")
        self.loading_label.grid(row=0, column=0, padx=20, pady=20)
        self.loading_label.grid_forget()  # Inicialmente escondido

        # Criar a área para exibir a imagem da webcam
        camera_frame = tk.Frame(self.root, bg="black")
        camera_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.label = tk.Label(camera_frame)
        self.label.pack(fill="both", expand=True)

        # Criar a área para exibir as informações
        self.info_text = tk.StringVar()
        self.info_text.set("Informações:\nNenhuma ação detectada.")
        self.info_label = tk.Label(self.root, textvariable=self.info_text, justify="left", font=("Helvetica", 14), anchor="w", width=40, relief="sunken", bg="#f4f4f9", height=10)
        self.info_label.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")
        self.info_label.grid_forget()  # Inicialmente escondido

        # Adicionar o botão de "Fechar"
        self.close_button = tk.Button(self.root, text="Fechar", command=self.close_window, font=("Helvetica", 14), relief="raised", width=20, bg="#FF5733", fg="white", activebackground="#ff3300", activeforeground="white")
        self.close_button.grid(row=2, column=0, padx=20, pady=20)
        self.close_button.grid_forget()  # Inicialmente escondido

    def run_detection(self):
        self.start_button.grid_forget()  # Esconde o botão de iniciar
        self.loading_label.grid(row=0, column=0, padx=20, pady=20)  # Exibe o label de carregamento
        self.loading_text = "ANIMUS"
        self.letter_index = 0  # Índice para as letras
        self.building_animation()  # Inicia a animação de construção do nome

        # Criar e iniciar o detector
        self.pose_detector = PoseDetector()
        threading.Thread(target=self.start_pose_detection, daemon=True).start()

        # Exibir o botão "Fechar" após iniciar a detecção
        self.close_button.grid(row=2, column=0, padx=20, pady=20)

    def building_animation(self):
        if self.letter_index < len(self.loading_text):
            # Atualiza o texto da label com a letra correspondente
            self.loading_label.config(text=self.loading_text[:self.letter_index + 1])
            self.loading_label.config(fg=self.get_next_color())  # Cor muda a cada letra
            self.letter_index += 1
            self.root.after(150, self.building_animation)  # Chama a função novamente após 150ms

    def get_next_color(self):
        colors = ["#FF5733", "#33FF57", "#3357FF", "#FF33A1", "#FF8C00"]
        return colors[self.letter_index % len(colors)]  # Cores alternando a cada letra

    def start_pose_detection(self):
        try:
            while True:
                mensagens, annotated_frame = self.pose_detector.detectar_pose()

                if mensagens:
                    self.info_text.set("\n".join(mensagens))
                else:
                    self.info_text.set("Nenhuma ação detectada.")

                # Mostrar informações e frame ao mesmo tempo
                self.info_label.grid(row=1, column=1, padx=20, pady=20, sticky="nsew")  # Exibe as informações

                # Converter a imagem e atualizar a interface com um pequeno atraso
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(annotated_frame_rgb)
                img = img.resize((640, 360))
                imgtk = ImageTk.PhotoImage(image=img)

                self.label.imgtk = imgtk
                self.label.configure(image=imgtk)

                # Atraso de 30ms para controlar o refresh da tela e evitar piscamento
                time.sleep(0.03)

                self.root.update()

        except Exception as e:
            self.pose_detector.liberarRecursos()
            self.loading_label.grid_forget()  # Esconde a animação
            self.info_text.set("Erro na detecção: " + str(e))

    def close_window(self):
        self.pose_detector.liberarRecursos()  # Libera os recursos da webcam
        self.root.quit()  # Fecha a aplicação

    def run(self):
        self.root.mainloop()

# Função principal para iniciar a interface gráfica
def main():
    gui = PoseDetectionGUI()
    gui.run()

if __name__ == "__main__":
    main()
