import cv2
import torch
import time
import numpy as np
from PIL import Image
from typing import Union
from transformers import AutoProcessor, AutoModelForCausalLM
import logging
import warnings

# Suppress the Florence2 generation warning
warnings.filterwarnings("ignore", message=".*Florence2LanguageForConditionalGeneration.*")


class LLMHandler:

    # Load Florence-2
    DEVICE = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")  # mps not supported yet

    def __init__(self, model_id: str = "microsoft/Florence-2-base-ft"):
        print(f"\n[LLM] Initializing Florence-2 model...")
        print(f"[LLM] Device: {self.DEVICE}")
        print(f"[LLM] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[LLM] CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"[LLM] CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        load_start = time.time()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True).to(self.DEVICE)
        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True)
        load_time = time.time() - load_start
        print(f"[LLM] Model loaded in {load_time:.2f}s")

    def inference_over_image(self, pil_image: Image, task_prompt: str):
        
        # Prepare input for Florence-2
        prep_start = time.time()
        inputs = self.processor(
            text=task_prompt, images=pil_image, return_tensors="pt").to(self.DEVICE)
        prep_time = time.time() - prep_start

        # Generate detection
        inference_start = time.time()
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                # do_sample=False,
                early_stopping=True
            )
        inference_time = time.time() - inference_start
        print(f"[LLM] Inference completed - Task: {task_prompt}, Prep: {prep_time:.3f}s, Inference: {inference_time:.3f}s")
        return generated_ids
    
    def describe_color(self, image: Union[np.ndarray, str], prompt: str):
        if "color" not in prompt:
            raise ValueError("Prompt must contain the word 'color'.")
        # Process the image
        if isinstance(image, str):
            image = cv2.imread(image)
        
        # Convert image to PIL
        # print("Input res: ", image.shape)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Infer over the image
        generated_ids = self.inference_over_image(pil_image, prompt)

        # Decode and process the response
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False)[0]

        # Print generated text for debugging
        # print("Generated text COLOR:", generated_text)

        # Buscar la etiqueta de color en la respuesta
        if 'The color of the' in generated_text:
            color = generated_text.split(
                'The color of the')[-1].split('is')[1].strip().split('.')[0]
        else:
            color = generated_text.strip()
        return color


    def describe_image(self, image: Union[np.ndarray, str], task_prompt: str):
        start_time = time.time()
        
        if isinstance(image, str):
            print(f"[LLM] Loading image from: {image}")
            image = cv2.imread(image)
            
        # Convert image to PIL
        print(f"[LLM] Processing image shape: {image.shape}, task: {task_prompt}")
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Infer over the image
        generated_ids = self.inference_over_image(pil_image, task_prompt)

        # Decode and process the response
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False)[0]

        # Process the response
        parsed_answer = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(pil_image.width, pil_image.height)
        )
        
        
        total_time = time.time() - start_time
        print(f"[LLM] Total time for describe_image: {total_time:.3f}s")

        return parsed_answer


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
