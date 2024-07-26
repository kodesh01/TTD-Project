import os
import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity

# Initialize MTCNN for face detection
mtcnn = MTCNN(keep_all=True)

# Initialize FaceNet model
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Directories to save captured face images
unknown_dir = r'C:\Users\kodesh\Music\facerecognition\unknown'
known_dir = r'C:\Users\kodesh\Music\facerecognition\known'
os.makedirs(unknown_dir, exist_ok=True)
os.makedirs(known_dir, exist_ok=True)

# Preprocessing function
def preprocess_face(face):
    """Preprocess the face image for the model."""
    face = cv2.resize(face, (160, 160))  # Resize face to 160x160
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    face = np.moveaxis(face, -1, 0)  # Change from HWC to CHW format
    face = np.expand_dims(face, axis=0).astype(np.float32)  # Add batch dimension and convert to float32
    face = (face / 255.0)  # Normalize to [0, 1]
    return face

def get_embedding(face):
    """Get face embedding from the model."""
    preprocessed_face = preprocess_face(face)
    with torch.no_grad():
        embedding = resnet(torch.from_numpy(preprocessed_face)).detach().numpy()
    return embedding

def detect_and_save_faces(frame):
    """Detect faces and save them after preprocessing."""
    global image_counter  # Declare image_counter as global to modify it
    global seen_faces  # Declare seen_faces as global to modify it

    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            
            if face.size == 0:
                continue  # Skip if no face is detected
            
            # Get face embedding
            embedding = get_embedding(face)
            
            # Compare embeddings
            recognized_as_known = False
            for known_embedding in seen_faces.values():
                similarity = cosine_similarity(embedding, known_embedding)
                if similarity >= 0.7:  # Adjust threshold as needed
                    recognized_as_known = True
                    break

            if recognized_as_known:
                # Face is recognized as known
                cv2.putText(frame, 'Known', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
                # Draw bounding box around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green bounding box
                image_filename = os.path.join(known_dir, f'face_{image_counter}.jpg')
            else:
                # New face
                cv2.putText(frame, 'Unknown', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)
                # Draw bounding box around face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding box
                image_filename = os.path.join(unknown_dir, f'face_{image_counter}.jpg')
                # Add the face embedding to seen faces
                seen_faces[f'face_{image_counter}'] = embedding
                image_counter += 1

            # Save the face image
            cv2.imwrite(image_filename, face)

# Load previously seen faces and their embeddings
seen_faces = {}
image_counter = 0  # Initialize counter for saving images

# Capture video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detect_and_save_faces(frame)

    # Display the frame
    cv2.imshow('Face Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
