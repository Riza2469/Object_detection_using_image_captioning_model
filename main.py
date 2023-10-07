import cv2
import torch
from PIL import Image
import pyttsx3
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor
import concurrent.futures

#model
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

#frame processing
def preprocess_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return image

#prediction
def predict_step(images):
    pixel_values = feature_extractor(images=images, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]

    return preds

#text-to-speech conversion
def text_to_speech(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.setProperty("volume", 1.0)
    engine.say(text)
    engine.runAndWait()


#video-capturing
def process_video():
    # Open the video capture - 0 for default webcam , 1 for other
    cap = cv2.VideoCapture(0)

    # Initialize an empty list to store the processed frames
    processed_frames = []

    # Flag to control speech output
    enable_speech = True

    # Loop over frames from the video stream
    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess the frame using multithreading
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(preprocess_frame, frame)
            image = future.result()

        processed_frames.append(image)

        # Call the predict_step function and pass the processed frames
        predictions = predict_step(image)
        print(predictions)

        # Output predictions as speech if enabled
        if enable_speech:
            for pred in predictions:
                text_to_speech(pred)

        # Display the frame with OpenCV
        cv2.imshow('Video', frame)

        #Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

    return predictions

process_video()


