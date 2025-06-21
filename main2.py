from flask import Flask, render_template, Response, jsonify, url_for, send_from_directory
import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from flask_cors import CORS
import pyttsx3
app = Flask(__name__)
CORS(app)

# Load the trained model
engine = pyttsx3.init()  
asana_sequence = ["Tadasana", "Tiryak_Tadasana_Left", "Tiryak_Tadasana_Right","Padahastasana","Ardha_Ustrasana","Pawanmuktasana","Ashwa_Sanchalanasana","Vajrasana"]
asanas = {
    'Marjariasana': 'Get on your hands and knees (tabletop position). Inhale, arch your back, and look up (cow pose). Exhale, round your spine, and tuck your chin to your chest (cat pose). Repeat.',
    'Shashankasana': 'Sit on your heels, stretch your arms forward, and lower your forehead to the ground. Keep your arms extended and relax your back.',
    'Padahastasana': 'Stand straight, exhale, and bend forward from the hips. Try to touch your toes or the ground with your hands. Keep your legs straight.',
    'Supta_Udarakarshanasana_Left': 'Lie on your back, bend your knees, and drop them to the left side while turning your head to the right. Keep your arms stretched out to the sides.',
    'Hasta_Uttanasana': 'Stand straight, raise your arms overhead, and stretch upward. Arch your back slightly and look up toward your hands.',
    'Savasana': 'Lie flat on your back, arms at your sides, palms facing up. Close your eyes and relax your entire body. Breathe naturally.',
    'Pawanmuktasana': 'Lie on your back, hug your knees to your chest, and lift your head to touch your knees. Hold the pose and breathe deeply.',
    'Saral_Bhujangasana': 'Lie on your stomach, place your palms under your shoulders, and gently lift your chest off the ground. Keep your elbows slightly bent.',
    'Baddha_Konasana': 'Sit with your legs stretched out. Bend your knees and bring the soles of your feet together. Hold your feet and gently flap your knees up and down.',
    'Supta_Udarakarshanasana_Right': 'Lie on your back, bend your knees, and drop them to the right side while turning your head to the left. Keep your arms stretched out to the sides.',
    'Ashwa_Sanchalanasana': 'From a standing position, step one leg back into a lunge. Keep your front knee bent at 90 degrees and stretch your back leg straight.',
    'Dandasana': 'Sit with your legs stretched out in front of you. Keep your spine straight, hands resting beside your hips, and toes pointing upward.',
    'Padottanasana': 'Stand with your feet wide apart. Bend forward from the hips and place your hands on the ground between your feet. Keep your legs straight.',
    'Vajrasana': 'Sit on your heels with your knees together. Keep your spine straight and place your hands on your thighs. Breathe deeply.',
    'Ardha_Ustrasana': 'Kneel on the ground, place your hands on your hips, and gently arch your back. Reach your hands toward your heels if possible.',
    'Tiryak_Tadasana_Right': 'Stand straight, raise your arms overhead, and interlock your fingers. Bend your upper body to the right side, keeping your feet grounded.',
    'Tiryak_Tadasana_Left': 'Stand straight, raise your arms overhead, and interlock your fingers. Bend your upper body to the left side, keeping your feet grounded.',
    'Tadasana': 'Stand straight with your feet together. Distribute your weight evenly on both feet. Keep your arms by your sides and your spine straight.'
}


# asana_sequence = [
#     "pranamasana",
#     "hasta_uttanasana",
#     "padahastasana",
#     "ashwa_sanchalanasana"
#     ,"adho_mukha_svanasana",

#     "ashtanga_namaskara",
#     "bhujangasana",
    
#     "ashwa_sanchalanasana",
#     "padahastasana",
#     "hasta_uttanasana",
#     "pranamasana"
# ]
hold_duration = 5     # Seconds to hold the pose
model_dict = pickle.load(open('yt_v2_model.p', 'rb'))
if(asana_sequence[0]=="pranamasana"):
    model_dict = pickle.load(open('/Users/joyboy/hack_frontend/suryanamaskar_model_1.p', 'rb'))
    hold_duration = 3
model = model_dict['model']
labels = model_dict['labels']



current_asana_index = 0
detection_duration = 3  # Seconds to detect correct pose
start_time = None
current_phase = "detecting"
session_completed = False
completed_asanas = [False] * len(asana_sequence)
timer_paused = False
pause_time = None

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Video Capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

def generate_frames():
    global current_asana_index, start_time, current_phase, session_completed, completed_asanas, timer_paused, pause_time

    while cap.isOpened() and not session_completed:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        H, W, _ = frame.shape
        predicted_asana = "None"
        show_detection = False

        if results.pose_landmarks and not session_completed:
            # Landmark processing
            x_ = [lm.x for lm in results.pose_landmarks.landmark]
            y_ = [lm.y for lm in results.pose_landmarks.landmark]
            data_aux = [coord for lm in results.pose_landmarks.landmark 
                      for coord in (lm.x - min(x_), lm.y - min(y_))]

            # Pose prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_asana = prediction[0]

            if predicted_asana == asana_sequence[current_asana_index]:
                if start_time is None:
                    start_time = current_time  # Start timer for the first time
                    timer_paused = False
                    pause_time = None

                if timer_paused:
                    # Resume timer if it was paused
                    start_time += (current_time - pause_time)
                    timer_paused = False
                    pause_time = None

                elapsed = current_time - start_time
                time_left = max(hold_duration - elapsed, 0)

                if time_left <= 0:
                    # Mark current asana as completed
                    completed_asanas[current_asana_index] = True

                    if current_asana_index < len(asana_sequence) - 1:
                        current_asana_index += 1
                        start_time = None  # Reset timer for next pose
                    else:
                        session_completed = True
                        cap.release()
                        cv2.destroyAllWindows()

                show_detection = True
            else:
                # Pause timer if user loses the correct pose
                if not timer_paused and start_time is not None:
                    timer_paused = True
                    pause_time = current_time

                show_detection = False

            # Draw annotations
            if show_detection:
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(frame, predicted_asana, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3)

            # Draw pose landmarks
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style()
            )

        # Stream frame
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return render_template('trainer_ai.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_image/<pose_name>')
def get_pose_image(pose_name):
    img_frmt=".jpeg"
    if(asana_sequence[0]=="Tadasana"):
        img_frmt=".png"
    try:
        return send_from_directory('static/images', f"{pose_name}{img_frmt}")
    except FileNotFoundError:
        return send_from_directory('static/images', 'placeholder.png')

@app.route('/get_status')
def get_status():
    current_time = time.time()
    time_left = 0

    if start_time is not None and not timer_paused:
        elapsed = current_time - start_time
        time_left = max(hold_duration - elapsed, 0)
    elif timer_paused and pause_time is not None:
        elapsed = pause_time - start_time
        time_left = max(hold_duration - elapsed, 0)
    current_pose = asana_sequence[current_asana_index] if not session_completed else None
    engine.say(asanas.get(current_pose, 'No description available.'))
    return jsonify({
        'current_pose': asana_sequence[current_asana_index] if not session_completed else None,
        'current_pose_image': url_for('get_pose_image', pose_name=current_pose) if current_pose else None,
        'current_phase': current_phase,
        'time_left': round(time_left, 1),
        'completed_poses': [asan for asan, completed in zip(asana_sequence, completed_asanas) if completed],
        'total_poses': len(asana_sequence),
        'progress': round((sum(completed_asanas)/len(asana_sequence)) * 100, 1),
        'is_completed': session_completed,
        'is_paused': timer_paused,
        'pose_description': asanas.get(current_pose, 'No description available.') if current_pose else None
    })
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)