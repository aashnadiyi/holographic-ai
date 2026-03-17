import sounddevice as sd
import speech_recognition as sr
import pyttsx3
import wavio
import cv2
import wikipedia
import numpy as np
import threading
import time

# ---------------- CONFIG ----------------
fs = 44100
seconds = 5
avatar_image = "avatar_face.jpg"  # Your avatar image

# ---------------- KENDALL JENNER AVATAR SPECIFICATIONS ----------------
AVATAR_SPECS = {
    # Body Proportions (in cm)
    "height": 178,
    "head_height": 23,
    "neck_circumference": 33,
    "shoulder_width": 41,
    "bust": 81,
    "waist": 61,
    "hips": 86,
    "torso_length": 41,
    "leg_length": 107,
    "thigh_circumference": 51,
    "calf_circumference": 36,
    "feet_length": 24,
    
    # Body Segment Ratios (for rigging)
    "segments": {
        "head": (0.0, 0.13),
        "neck_shoulders": (0.13, 0.20),
        "chest": (0.20, 0.30),
        "waist": (0.30, 0.37),
        "hips": (0.37, 0.45),
        "thighs": (0.45, 0.65),
        "knees": (0.65, 0.75),
        "calves": (0.75, 0.90),
        "feet": (0.90, 1.0)
    },
    
    # Colors (RGB)
    "colors": {
        "skin": (244, 227, 218),
        "hair": (55, 38, 29),
        "eyes": (62, 43, 36),
        "lips": (193, 134, 129),
        "eyebrows": (55, 38, 29)
    },
    
    # Facial Features
    "face": {
        "shape": "oval",
        "forehead_ratio": 1/3,
        "nose_ratio": 1/3,
        "lips_ratio": 1/5,
        "eyes_width_ratio": 1/5,
        "eyebrow_distance": 0.5  # cm above eyes
    },
    
    # Hair
    "hair": {
        "length": 60,  # cm
        "style": "straight",
        "part": "center",
        "texture": "smooth"
    }
}

# ---------------- INIT ----------------
recognizer = sr.Recognizer()
is_speaking = False

# ---------------- MODEL-QUALITY AVATAR ----------------
class ModelAvatar:
    def __init__(self, image_path):
        self.specs = AVATAR_SPECS
        self.base_image = cv2.imread(image_path)
        if self.base_image is None:
            self.base_image = self.create_model_avatar()
        self.height, self.width = self.base_image.shape[:2]

    def create_model_avatar(self):
        """Create a model-quality avatar based on specifications"""
        # Scale factor for screen (178cm height -> 900 pixels)
        scale = 900 / self.specs["height"]
        
        img = np.ones((900, 600, 3), dtype=np.uint8) * 255
        
        # Get colors
        skin = self.specs["colors"]["skin"]
        hair_color = self.specs["colors"]["hair"]
        eye_color = self.specs["colors"]["eyes"]
        lip_color = self.specs["colors"]["lips"]
        
        # Calculate body segment positions (y-coordinates)
        seg = self.specs["segments"]
        head_y = int(seg["head"][1] * 900)
        neck_y = int(seg["neck_shoulders"][1] * 900)
        chest_y = int(seg["chest"][1] * 900)
        waist_y = int(seg["waist"][1] * 900)
        hips_y = int(seg["hips"][1] * 900)
        thighs_y = int(seg["thighs"][1] * 900)
        knees_y = int(seg["knees"][1] * 900)
        calves_y = int(seg["calves"][1] * 900)
        
        center_x = 300
        
        # ==== HEAD - Oval face shape ====
        head_width = int(self.specs["shoulder_width"] * scale * 0.7)
        head_height = int(self.specs["head_height"] * scale)
        
        # Oval face
        cv2.ellipse(img, (center_x, int(head_y * 0.7)), 
                   (head_width, head_height), 0, 0, 360, skin, -1)
        
        # Face contour shading
        cv2.ellipse(img, (center_x - 40, int(head_y * 0.8)), 
                   (15, 30), 0, 0, 360, (234, 217, 208), -1)
        cv2.ellipse(img, (center_x + 40, int(head_y * 0.8)), 
                   (15, 30), 0, 0, 360, (234, 217, 208), -1)
        
        # ==== HAIR - Dark brown, long, straight ====
        hair_length = int(self.specs["hair"]["length"] * scale)
        
        # Back hair (long, flowing)
        cv2.ellipse(img, (center_x, 80), (head_width + 20, head_height + 30), 
                   0, 0, 360, hair_color, -1)
        
        # Long side strands
        left_hair = np.array([
            [center_x - head_width - 10, 100],
            [center_x - head_width - 20, 200],
            [center_x - head_width - 15, 300],
            [center_x - head_width, 250],
            [center_x - head_width + 10, 150]
        ], np.int32)
        cv2.fillPoly(img, [left_hair], hair_color)
        
        right_hair = np.array([
            [center_x + head_width + 10, 100],
            [center_x + head_width + 20, 200],
            [center_x + head_width + 15, 300],
            [center_x + head_width, 250],
            [center_x + head_width - 10, 150]
        ], np.int32)
        cv2.fillPoly(img, [right_hair], hair_color)
        
        # Top hair
        cv2.ellipse(img, (center_x, 60), (head_width + 10, 50), 
                   0, 0, 360, hair_color, -1)
        
        # Center part
        cv2.line(img, (center_x, 30), (center_x, 90), (45, 33, 24), 2)
        
        # ==== EYES - Almond shaped, dark brown ====
        eye_y = int(head_y * 0.65)
        eye_distance = int(head_width * 0.35)
        
        # Left eye
        left_eye_pts = np.array([
            [center_x - eye_distance - 25, eye_y],
            [center_x - eye_distance + 5, eye_y - 5],
            [center_x - eye_distance + 25, eye_y],
            [center_x - eye_distance + 5, eye_y + 8]
        ], np.int32)
        cv2.fillPoly(img, [left_eye_pts], (255, 255, 255))
        
        # Left iris
        cv2.ellipse(img, (center_x - eye_distance, eye_y + 2), 
                   (12, 16), 0, 0, 360, eye_color, -1)
        cv2.circle(img, (center_x - eye_distance, eye_y + 4), 6, (20, 15, 12), -1)
        cv2.circle(img, (center_x - eye_distance - 3, eye_y - 2), 4, (255, 255, 255), -1)
        
        # Right eye
        right_eye_pts = np.array([
            [center_x + eye_distance - 25, eye_y],
            [center_x + eye_distance - 5, eye_y - 5],
            [center_x + eye_distance + 25, eye_y],
            [center_x + eye_distance - 5, eye_y + 8]
        ], np.int32)
        cv2.fillPoly(img, [right_eye_pts], (255, 255, 255))
        
        # Right iris
        cv2.ellipse(img, (center_x + eye_distance, eye_y + 2), 
                   (12, 16), 0, 0, 360, eye_color, -1)
        cv2.circle(img, (center_x + eye_distance, eye_y + 4), 6, (20, 15, 12), -1)
        cv2.circle(img, (center_x + eye_distance + 3, eye_y - 2), 4, (255, 255, 255), -1)
        
        # Eyeliner and lashes
        cv2.ellipse(img, (center_x - eye_distance, eye_y - 2), 
                   (26, 8), 0, 180, 360, (30, 25, 20), 2)
        cv2.ellipse(img, (center_x + eye_distance, eye_y - 2), 
                   (26, 8), 0, 180, 360, (30, 25, 20), 2)
        
        # Eyebrows - straight with soft arch
        brow_y = eye_y - 18
        left_brow = np.array([
            [center_x - eye_distance - 28, brow_y + 3],
            [center_x - eye_distance - 10, brow_y],
            [center_x - eye_distance + 10, brow_y + 2]
        ], np.int32)
        right_brow = np.array([
            [center_x + eye_distance - 10, brow_y + 2],
            [center_x + eye_distance + 10, brow_y],
            [center_x + eye_distance + 28, brow_y + 3]
        ], np.int32)
        cv2.polylines(img, [left_brow], False, self.specs["colors"]["eyebrows"], 3)
        cv2.polylines(img, [right_brow], False, self.specs["colors"]["eyebrows"], 3)
        
        # ==== NOSE - Straight, narrow ====
        nose_y = int(head_y * 0.8)
        cv2.line(img, (center_x, eye_y + 20), (center_x, nose_y), (234, 217, 208), 1)
        cv2.ellipse(img, (center_x - 5, nose_y), (4, 3), 0, 0, 180, (234, 217, 208), 1)
        cv2.ellipse(img, (center_x + 5, nose_y), (4, 3), 0, 0, 180, (234, 217, 208), 1)
        
        # High cheekbones
        cv2.ellipse(img, (center_x - 45, nose_y - 10), (18, 12), -20, 0, 360, (237, 222, 213), -1)
        cv2.ellipse(img, (center_x + 45, nose_y - 10), (18, 12), 20, 0, 360, (237, 222, 213), -1)
        
        # ==== NECK - Slender (33cm circumference) ====
        neck_width = int(self.specs["neck_circumference"] * scale * 0.4)
        neck_pts = np.array([
            [center_x - neck_width, head_y],
            [center_x - neck_width - 3, neck_y],
            [center_x + neck_width + 3, neck_y],
            [center_x + neck_width, head_y]
        ], np.int32)
        cv2.fillPoly(img, [neck_pts], skin)
        
        # ==== SHOULDERS - 41cm width ====
        shoulder_width = int(self.specs["shoulder_width"] * scale)
        shoulder_pts = np.array([
            [center_x - neck_width - 3, neck_y],
            [center_x - shoulder_width, neck_y + 20],
            [center_x - shoulder_width + 10, chest_y],
            [center_x - 40, chest_y]
        ], np.int32)
        cv2.fillPoly(img, [shoulder_pts], skin)
        
        shoulder_pts_r = np.array([
            [center_x + neck_width + 3, neck_y],
            [center_x + shoulder_width, neck_y + 20],
            [center_x + shoulder_width - 10, chest_y],
            [center_x + 40, chest_y]
        ], np.int32)
        cv2.fillPoly(img, [shoulder_pts_r], skin)
        
        # ==== BODY - Bust: 81cm, Waist: 61cm, Hips: 86cm ====
        bust_width = int(self.specs["bust"] * scale * 0.25)
        waist_width = int(self.specs["waist"] * scale * 0.25)
        hip_width = int(self.specs["hips"] * scale * 0.25)
        
        # Torso with curves
        body_outline = np.array([
            # Chest
            [center_x - bust_width, chest_y],
            [center_x - bust_width + 5, chest_y + 20],
            # Waist
            [center_x - waist_width, waist_y],
            [center_x - waist_width - 5, waist_y + 10],
            # Hips
            [center_x - hip_width, hips_y],
            [center_x - hip_width + 5, thighs_y],
            # Thighs
            [center_x - 35, knees_y],
            # Calves
            [center_x - 25, calves_y],
            # Center
            [center_x, calves_y + 20],
            # Mirror right side
            [center_x + 25, calves_y],
            [center_x + 35, knees_y],
            [center_x + hip_width - 5, thighs_y],
            [center_x + hip_width, hips_y],
            [center_x + waist_width + 5, waist_y + 10],
            [center_x + waist_width, waist_y],
            [center_x + bust_width - 5, chest_y + 20],
            [center_x + bust_width, chest_y]
        ], np.int32)
        
        cv2.fillPoly(img, [body_outline], (250, 245, 245))
        
        # Body contours for definition
        cv2.line(img, (center_x, chest_y), (center_x, waist_y), (240, 230, 225), 1)
        cv2.line(img, (center_x - waist_width + 2, waist_y), 
                (center_x - waist_width + 5, waist_y + 30), (240, 230, 225), 1)
        cv2.line(img, (center_x + waist_width - 2, waist_y), 
                (center_x + waist_width - 5, waist_y + 30), (240, 230, 225), 1)
        
        # Simple clothing
        # Top (white)
        cv2.polylines(img, [body_outline[:10]], False, (200, 200, 200), 2)
        
        return img

    def add_lips(self, image, openness=0):
        """Add full, natural pink lips based on specs"""
        img = image.copy()
        mouth_x = self.width // 2
        mouth_y = int(self.height * 0.125)  # Position for lips on face
        
        lip_color = self.specs["colors"]["lips"]
        lip_outline = (173, 114, 109)
        
        if openness < 0.15:
            # Closed - full natural lips
            upper_lip = np.array([
                [mouth_x - 30, mouth_y],
                [mouth_x - 20, mouth_y - 4],
                [mouth_x - 5, mouth_y - 3],
                [mouth_x, mouth_y - 5],
                [mouth_x + 5, mouth_y - 3],
                [mouth_x + 20, mouth_y - 4],
                [mouth_x + 30, mouth_y]
            ], np.int32)
            cv2.fillPoly(img, [upper_lip], lip_color)
            cv2.polylines(img, [upper_lip], False, lip_outline, 2)
            
            lower_lip = np.array([
                [mouth_x - 30, mouth_y],
                [mouth_x - 15, mouth_y + 6],
                [mouth_x, mouth_y + 8],
                [mouth_x + 15, mouth_y + 6],
                [mouth_x + 30, mouth_y]
            ], np.int32)
            cv2.fillPoly(img, [lower_lip], lip_color)
            cv2.polylines(img, [lower_lip], False, lip_outline, 2)
            
            # Lip highlight
            cv2.ellipse(img, (mouth_x, mouth_y + 4), (8, 3), 0, 0, 180, (220, 180, 175), -1)
            
        elif openness < 0.4:
            # Small open
            cv2.ellipse(img, (mouth_x, mouth_y + 4), (20, 16), 0, 0, 360, lip_color, -1)
            cv2.ellipse(img, (mouth_x, mouth_y + 4), (20, 16), 0, 0, 360, lip_outline, 2)
            cv2.ellipse(img, (mouth_x, mouth_y + 5), (15, 12), 0, 0, 360, (255, 200, 200), -1)
            cv2.ellipse(img, (mouth_x, mouth_y + 5), (12, 9), 0, 0, 360, (60, 40, 35), -1)
            
        elif openness < 0.6:
            # Medium open - speaking
            cv2.ellipse(img, (mouth_x, mouth_y), (24, 9), 0, 0, 180, lip_color, -1)
            cv2.ellipse(img, (mouth_x, mouth_y), (24, 9), 0, 0, 180, lip_outline, 2)
            
            cv2.ellipse(img, (mouth_x, mouth_y + 16), (24, 9), 0, 180, 360, lip_color, -1)
            cv2.ellipse(img, (mouth_x, mouth_y + 16), (24, 9), 0, 180, 360, lip_outline, 2)
            
            cv2.ellipse(img, (mouth_x, mouth_y + 8), (20, 13), 0, 0, 360, (60, 40, 35), -1)
            cv2.rectangle(img, (mouth_x - 16, mouth_y + 2), (mouth_x + 16, mouth_y + 6), (255, 255, 255), -1)
            
        else:
            # Wide open
            cv2.ellipse(img, (mouth_x, mouth_y + 2), (28, 11), 0, 0, 180, lip_color, -1)
            cv2.ellipse(img, (mouth_x, mouth_y + 2), (28, 11), 0, 0, 180, lip_outline, 2)
            
            cv2.ellipse(img, (mouth_x, mouth_y + 22), (28, 11), 0, 180, 360, lip_color, -1)
            cv2.ellipse(img, (mouth_x, mouth_y + 22), (28, 11), 0, 180, 360, lip_outline, 2)
            
            cv2.ellipse(img, (mouth_x, mouth_y + 12), (25, 19), 0, 0, 360, (50, 30, 25), -1)
            cv2.ellipse(img, (mouth_x, mouth_y + 5), (22, 6), 0, 0, 180, (255, 255, 255), -1)
            cv2.ellipse(img, (mouth_x, mouth_y + 18), (20, 5), 0, 180, 360, (255, 255, 255), -1)
        
        return img

    def add_animations(self, image, frame_num):
        """Add subtle model-like movements"""
        # Very subtle head tilt
        angle = np.sin(frame_num * 0.03) * 1.2
        
        center = (self.width // 2, self.height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Gentle sway
        matrix[0, 2] += np.sin(frame_num * 0.025) * 1.5
        matrix[1, 2] += np.cos(frame_num * 0.04) * 1.8
        
        result = cv2.warpAffine(image, matrix, (self.width, self.height),
                               borderMode=cv2.BORDER_REPLICATE)
        
        # Occasional blink
        if frame_num % 180 == 0 or frame_num % 180 == 1:
            cv2.line(result, (235, 85), (265, 85), (244, 227, 218), 3)
            cv2.line(result, (335, 85), (365, 85), (244, 227, 218), 3)
        
        return result

# ---------------- TEXT TO MOUTH SHAPES ----------------
def text_to_mouth_shapes(text):
    shapes = []
    vowel_open = {'a': 0.9, 'e': 0.7, 'i': 0.5, 'o': 0.85, 'u': 0.6}
    consonant_open = {
        'm': 0.0, 'n': 0.0, 'b': 0.0, 'p': 0.0,
        'f': 0.3, 'v': 0.3, 's': 0.4, 'z': 0.4,
        't': 0.2, 'd': 0.2, 'k': 0.3, 'g': 0.3,
        'l': 0.4, 'r': 0.5, 'w': 0.6, 'y': 0.5,
        'h': 0.5, 'j': 0.4, 'q': 0.3, 'x': 0.4, 'c': 0.3
    }
    
    for char in text.lower():
        if char in vowel_open:
            shapes.append((0.12, vowel_open[char]))
        elif char in consonant_open:
            shapes.append((0.08, consonant_open[char]))
        elif char == ' ':
            shapes.append((0.1, 0.1))
        elif char in '.!?':
            shapes.append((0.15, 0.05))
        else:
            shapes.append((0.06, 0.3))
    
    return shapes

# ---------------- SPEAK WITH LIP SYNC ----------------
def speak_with_lipsync(text, avatar):
    global is_speaking
    print("LUMA:", text)
    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    mouth_shapes = text_to_mouth_shapes(text)

    def speak_thread():
        engine.say(text)
        engine.runAndWait()

    is_speaking = True
    thread = threading.Thread(target=speak_thread)
    thread.start()

    frame_num = 0
    shape_index = 0
    
    while thread.is_alive() and shape_index < len(mouth_shapes):
        duration, openness = mouth_shapes[shape_index]
        
        animated_image = avatar.add_animations(avatar.base_image, frame_num)
        final_image = avatar.add_lips(animated_image, openness)
        
        cv2.imshow("LUMA Hologram", final_image)
        cv2.waitKey(int(duration * 1000))
        
        frame_num += 1
        shape_index += 1

    final_image = avatar.add_lips(avatar.base_image, 0)
    cv2.imshow("LUMA Hologram", final_image)
    
    thread.join()
    is_speaking = False

# ---------------- LISTEN ----------------
def listen_audio(seconds=5):
    print("🎤 Listening...")
    recording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    wavio.write("temp.wav", recording, fs, sampwidth=2)
    
    with sr.AudioFile("temp.wav") as source:
        audio = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print("You said:", text)
        return text.lower()
    except:
        return None

# ---------------- THINK ----------------
def think(query):
    if "capital of telangana" in query:
        return "Hyderabad is the capital of Telangana."
    if "india" in query:
        return "New Delhi is the capital of India."
    try:
        return wikipedia.summary(query, sentences=1)
    except:
        return "I'm still learning about that."

# ---------------- MAIN ----------------
def main():
    avatar = ModelAvatar(avatar_image)
    speak_with_lipsync("Hello Aashna. I am LUMA, your holographic assistant.", avatar)
    
    frame_num = 0

    while True:
        if not is_speaking:
            idle_img = avatar.add_animations(avatar.base_image, frame_num)
            idle_img = avatar.add_lips(idle_img, 0)
            
            cv2.imshow("LUMA Hologram", idle_img)
            
            key = cv2.waitKey(50)
            frame_num += 1
            
            if key == ord('q'):
                break

            user_text = listen_audio()
            if user_text:
                if "exit" in user_text or "bye" in user_text:
                    speak_with_lipsync("Goodbye Aashna.", avatar)
                    break
                
                answer = think(user_text)
                speak_with_lipsync(answer, avatar)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()