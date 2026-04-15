import tkinter as tk
import cv2
import logging
import os
from PIL import Image, ImageTk, ImageDraw 
import subprocess
from tkinter import filedialog
from tkinter import messagebox
import time
# Configure logging
logging.basicConfig(filename="detection.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Function to run a Python script when the button is clicked
def run_script(script_path="app.py"):
    try:
        subprocess.run(["python", script_path], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
    except FileNotFoundError:
        messagebox.showerror("Error", f"Script file '{script_path}' not found.")

# Function to create a rounded rectangle button with text
def create_rounded_button(canvas, x, y, width, height, radius, text, font, fill_color, text_color):
    rect_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(rect_image)
    draw.rounded_rectangle((0, 0, width, height), radius=radius, fill=fill_color)
    rect_photo = ImageTk.PhotoImage(rect_image)
    
    rect_id = canvas.create_image(x, y, image=rect_photo, anchor="center")
    text_id = canvas.create_text(x, y, text=text, font=font, fill=text_color)
    
    if not hasattr(canvas, "image_references"):
        canvas.image_references = []
    canvas.image_references.append(rect_photo)
    return rect_id, text_id

# Function to detect age and gender in an image
def detect_age_gender(image_path):
    try:
        # Load models
        faceProto = "opencv_face_detector.pbtxt"
        faceModel = "opencv_face_detector_uint8.pb"
        ageProto = "age_deploy.prototxt"
        ageModel = "age_net.caffemodel"
        genderProto = "gender_deploy.prototxt"
        genderModel = "gender_net.caffemodel"

        faceNet = cv2.dnn.readNet(faceModel, faceProto)
        ageNet = cv2.dnn.readNet(ageModel, ageProto)
        genderNet = cv2.dnn.readNet(genderModel, genderProto)

        # Enable GPU if available, otherwise fallback to CPU
        try:
            faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        except:
            print("CUDA not available, falling back to CPU.")
            faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            ageNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            ageNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            genderNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            genderNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60+)']
        genderList = ['Male', 'Female']

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError(f"Could not read image from {image_path}.")

        padding = 20
        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            print("No face detected")
            return

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                         max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            label = f'{gender}, {age}'
            print(label)
            logging.info(f"Detected: {label}")
            cv2.putText(resultImg, label, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Age and Gender Detection", resultImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        messagebox.showerror("Error", str(e))

# Function to highlight faces in the frame
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), 2, 8)
    return frameOpencvDnn, faceBoxes

# Create the main application window
root = tk.Tk()
root.title("Age & Gender Detection Program")
root.geometry("800x600")
root.resizable(True, True)

# Load and display the background image
try:
    bg_image = Image.open("_internal/age2.jpg")  # Replace with your image file path
    bg_image = bg_image.resize((800, 600), Image.LANCZOS)
    bg_photo = ImageTk.PhotoImage(bg_image)
except Exception as e:
    messagebox.showerror("Error", f"Failed to load background image: {e}")
    bg_photo = None

# Create a canvas to hold the background image
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack(fill="both", expand=True)
if bg_photo:
    canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Add title text directly to the canvas
canvas.create_text(400, 100, text="Age & Gender Detection Program", font=("Arial", 24, "bold"), fill="White")

# Add instruction text directly to the canvas
canvas.create_text(400, 250, text="Click below to Run the Program and press 'q' to Exit", font=("Arial", 14), fill="White")

# Create the first button
button_id_1, button_text_1 = create_rounded_button(
    canvas, x=400, y=320, width=200, height=50, radius=25, 
    text="Live Detect", font=("Arial", 18, "bold"), fill_color="#FF5733", text_color="White"
)

# Bind click event to the first button
def on_button_click_1(event):
    run_script("app.py")

canvas.tag_bind(button_id_1, "<Button-1>", on_button_click_1)
canvas.tag_bind(button_text_1, "<Button-1>", on_button_click_1)

# Add another instruction text directly to the canvas
canvas.create_text(400, 400, text="Click the button below to Upload an image", font=("Arial", 14), fill="White")

# Create the second button
button_id_2, button_text_2 = create_rounded_button(
    canvas, x=400, y=470, width=200, height=50, radius=25, 
    text="Image Detection", font=("Arial", 18, "bold"), fill_color="#28A745", text_color="white"
)

# Bind click event to the second button
def on_button_click_2(event):
    file_path = filedialog.askopenfilename(
        title="Select an Image File", 
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.gif;*.bmp")]
    )
    if file_path:
        detect_age_gender(file_path)

canvas.tag_bind(button_id_2, "<Button-1>", on_button_click_2)
canvas.tag_bind(button_text_2, "<Button-1>", on_button_click_2)

# Run the application
root.mainloop()