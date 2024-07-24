import cv2
import torch
import time
import numpy as np
from PIL import Image
from typing import Union
from transformers import AutoProcessor, AutoModelForCausalLM


class LLMHandler:

    # Cargar Florence-2
    CHECKPOINT = "microsoft/Florence-2-base-ft"
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # mps not supported yet

    def __init__(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.CHECKPOINT, trust_remote_code=True).to(self.DEVICE)
        self.processor = AutoProcessor.from_pretrained(
            self.CHECKPOINT, trust_remote_code=True)

    def describe_image(self, image: Union[np.ndarray, str], task_prompt: str):

        if isinstance(image, str):
            image = cv2.imread(image)
        # Convertir imagen a PIL
        # print("Input res: ", image.shape)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Preparar entrada para Florence-2
        inputs = self.processor(
            text=task_prompt, images=pil_image, return_tensors="pt").to(self.DEVICE)

        # Generar detección de color
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                # do_sample=False,
                early_stopping=True
            )

        # Decodificar y procesar la respuesta
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        if "color" in task_prompt:
            # Imprimir el texto generado para depuración
            print("Generated text COLOR:", generated_text)

            # Buscar la etiqueta de color en la respuesta
            if 'The color of the' in generated_text:
                color = generated_text.split(
                    'The color of the')[-1].split('is')[1].strip().split('.')[0]
            else:
                color = generated_text.strip()
            return color
        
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(pil_image.width, pil_image.height)
        )
            
        return parsed_answer

    def describe_image_on_crop(self, name,  box, image_path, task_prompt: str, aux_prompt: Union[str, None] = None):

        aux_result = None
        # Obtener el objeto del frame
        image = cv2.imread(image_path)
        if image is None:
            print(f"Image not found at {image_path}.")
            return

        # Recortar el objeto de la imagen
        x1, y1, x2, y2 = map(int, [box['x1'], box['y1'], box['x2'], box['y2']])
        cropped_image = image[y1:y2, x1:x2]

        # Aumentar el tamaño de la imagen recortada para una mejor visualización
        resized_cropped_image = cv2.resize(cropped_image, (300, 300))

        # Describe object
        result = self.describe_image(resized_cropped_image, task_prompt)

        if "color" in task_prompt:
            output_result_test = f"The predominant color of the {name} is {result}."
            return output_result_test

        
        if aux_prompt is not None:
            aux_result = self.describe_image(resized_cropped_image, aux_prompt)
    
        return {
            "result_aux": aux_result,
            "result_task": result
        }

if __name__ == '__main__':
    llm = LLMHandler()
    print("Startingg...")

    # Test with video input from webcam

    cap = cv2.VideoCapture(0)
    while True:
        time1 = time.time()
        ret, frame = cap.read()
        
        # REsize
        # frame = cv2.resize(frame, (320, 240))
        if not ret:
            break
        cv2.imshow("frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print("Processing frame...")
        # prompt = "<OD>"
        # prompt = "<DENSE_REGION_CAPTION>"
        # prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        prompt = "<MORE_DETAILED_CAPTION>"
        result = llm.describe_image(frame,  prompt)
        print("Result: ", result)
        time2 = time.time()
        
        # Print FPS
        print("FPS: , Time (ms)", 1/(time2-time1), (time2-time1)*1000)
