import os
import threading
import glob
import platform
import subprocess
import cv2
import numpy as np
import time
from sklearn.cluster import KMeans
from nanpy import ArduinoApi, SerialManager, Servo
from time import sleep
from flask import Flask, redirect, url_for, render_template, request, jsonify, Response
from picamera2 import Picamera2
from ultralytics import YOLO
from sklearn.cluster import KMeans

app = Flask(__name__)

# =======================
# Global Config
# =======================
model = YOLO('/home/admin/Desktop/thesisFlask/static/model/best.pt')  # YOLO model

captured_dir = '/home/admin/Desktop/thesisFlask/static/captured_images'
predicted_dir = '/home/admin/Desktop/thesisFlask/static/predicted_images'
os.makedirs(predicted_dir, exist_ok=True)


# Global variables
size_matrix = None
servos = []
moved_servos = []
processed_sizes = set()
boot_status = {
    "progress": 0,
    "message": "Starting..."
}


# =======================
# Servo Functions
# =======================
def initialize_servos():
    global servos, boot_status
    try:
        boot_status["message"] = "Connecting to Arduino 1..."
        boot_status["progress"] = 20
        connection1 = SerialManager(device='/dev/arduino1')
        arduino1 = ArduinoApi(connection1)

        boot_status["message"] = "Connecting to Arduino 2..."
        boot_status["progress"] = 40
        connection2 = SerialManager(device='/dev/arduino2')
        arduino2 = ArduinoApi(connection2)

        servos_local = []

        boot_status["message"] = "Initializing Arduino 1 servos..."
        boot_status["progress"] = 60
        for i in range(13, 1, -1):
            servo = Servo(pin=i, connection=connection1)
            servo.write(0)
            sleep(0.05)
            servos_local.append(servo)
        for i in range(44, 47):
            servo = Servo(pin=i, connection=connection1)
            servo.write(0)
            sleep(0.05)
            servos_local.append(servo)

        boot_status["message"] = "Initializing Arduino 2 servos..."
        boot_status["progress"] = 80
        for i in range(46, 43, -1):
            servo = Servo(pin=i, connection=connection2)
            servo.write(0)
            sleep(0.05)
            servos_local.append(servo)
        for i in range(2, 14):
            servo = Servo(pin=i, connection=connection2)
            servo.write(0)
            sleep(0.05)
            servos_local.append(servo)

        servos = servos_local
        boot_status["message"] = f"{len(servos)} servos initialized."
        boot_status["progress"] = 100
        print(f"{len(servos)} servos initialized.")
        return servos
    except Exception as e:
        boot_status["message"] = f"Error: {e}"
        boot_status["progress"] = 100
        print(f"Error initializing servos: {e}")
        return None




def actuate_size(size_to_actuate):
    global moved_servos, size_matrix, servos
    if size_matrix is None:
        print("Error: size_matrix not set.")
        return
    batch_servos = []
    for row in range(5):
        for col in range(6):
            # Apply 180° rotation mapping
            row_idx = 4 - row
            col_idx = 5 - col
            idx = row_idx * 6 + col_idx

            if idx >= len(servos):
                continue
            if size_matrix[row, col] == size_to_actuate:
                batch_servos.append(servos[idx])
    if not batch_servos:
        print(f"No servos found for size '{size_to_actuate}'")
        return
    print(f"Actuating {len(batch_servos)} servos for '{size_to_actuate}'")
    for s in batch_servos:
        s.write(0)
    sleep(1.5)
    for s in batch_servos:
        s.write(90)
    sleep(0.8)
    for s in batch_servos:
        s.write(0)
    sleep(1.5)
    for s in batch_servos:
        if s not in moved_servos:
            moved_servos.append(s)
    print(f"Finished actuating '{size_to_actuate}'")




def reset_moved_servos():
    global moved_servos
    if not moved_servos:
        return
    print("Resetting previously moved servos...")
    for s in moved_servos:
        s.write(0)
    sleep(1.5)
    for s in moved_servos:
        s.write(90)
    sleep(0.8)
    for s in moved_servos:
        s.write(0)
    sleep(1.5)
    moved_servos.clear()
    print("Reset done.")

def reset_all_servos():
    global servos
    print("Resetting all servos...")
    for s in servos:
        s.write(0)
    sleep(1.5)
    print("All servos reset.")

# Initialize servos at startup
@app.before_first_request
def startup():
    threading.Thread(target=initialize_servos, daemon=True).start()

# =======================
# Initialize ONE Camera
# =======================
picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1920, 1080)},  
    controls={"FrameDurationLimits": (33333, 33333)}  
)
picam2.configure(config)

controls = {
    "AeEnable": False,
    "ExposureTime": 1250,
    "AnalogueGain": 5.0,
    "AwbEnable": True,
    "NoiseReductionMode": 2,
}
picam2.set_controls(controls)
picam2.start()
sleep(2)  # stabilize camera

# =======================
# Utility Functions
# =======================
def gen_frames():
    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    except GeneratorExit:
        pass

def capture_and_save_image():
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    existing_files = [f for f in os.listdir(captured_dir) if f.endswith(".jpg")]
    existing_numbers = [int(os.path.splitext(f)[0]) for f in existing_files if os.path.splitext(f)[0].isdigit()]
    next_number = max(existing_numbers, default=0) + 1
    image_filename = f"{next_number}.jpg"
    save_path = os.path.join(captured_dir, image_filename)
    cv2.imwrite(save_path, gray_frame)
    print(f"Image saved: {save_path}")
    return save_path

def get_latest_image():
    files = glob.glob(os.path.join(captured_dir, '*.jpg'))
    return max(files, key=os.path.getctime) if files else None

# =======================
# Size Matrix Extraction (exact working code)
# =======================
def extract_size_matrix(image_path, model_path='/home/admin/Desktop/thesisFlask/static/model/best.pt'):
    """
    Runs YOLO detection on the captured image and builds a 2D egg tray size matrix.
    """
    from ultralytics import YOLO
    import numpy as np

    model = YOLO(model_path)
    results = model.predict(image_path, conf=0.49)
    result = results[0]

    tray_xmin, tray_ymin, tray_xmax, tray_ymax = (350, 0, 1651, 1079)
    tray_w = tray_xmax - tray_xmin
    tray_h = tray_ymax - tray_ymin

    rows, cols = (5, 6)
    grid = np.full((rows, cols), "empty", dtype=object)
    cell_w = tray_w / cols
    cell_h = tray_h / rows

    detected_sizes = set()  # 🔑 Collect unique detected sizes

    for box in result.boxes:
        x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
        cls = int(box.cls[0].item())
        label = result.names[cls]  # s, m, l, xl

        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2

        if not (tray_xmin <= cx <= tray_xmax and tray_ymin <= cy <= tray_ymax):
            continue

        col_idx = int((cx - tray_xmin) // cell_w)
        row_idx = int((cy - tray_ymin) // cell_h)

        if 0 <= row_idx < rows and 0 <= col_idx < cols:
            grid[row_idx, col_idx] = label
            detected_sizes.add(label)

    return grid, results, list(detected_sizes)



def reset_processed_servos():
    """
    Reset all servos corresponding to processed sizes simultaneously.
    """
    global processed_sizes, size_matrix, servos

    if not processed_sizes or size_matrix is None:
        print("No processed sizes to reset.")
        return False, "No processed sizes to reset."

    print(f"Resetting servos for processed sizes: {processed_sizes}")

    try:
        servos_to_reset = []
        for size in list(processed_sizes):
            for row in range(5):
                for col in range(6):
                    # Apply 180° rotation mapping
                    row_idx = 4 - row
                    col_idx = 5 - col
                    idx = row_idx * 6 + col_idx

                    if idx >= len(servos):
                        continue
                    if size_matrix[row, col] == size:
                        servos_to_reset.append(servos[idx])

        if not servos_to_reset:
            return False, "No servos found for processed sizes."

        for s in servos_to_reset:
            s.write(0)
        time.sleep(1.5)

        for s in servos_to_reset:
            s.write(90)
        time.sleep(0.8)

        for s in servos_to_reset:
            s.write(0)
        time.sleep(1.5)

        processed_sizes.clear()
        print("Processed sizes reset successfully (simultaneous).")
        return True, "Processed sizes reset successfully."
    except Exception as e:
        return False, str(e)



# =======================
# Routes
# =======================
@app.route("/status")
def status():
    global boot_status
    return jsonify(boot_status)


@app.route("/")
def idle():
    return render_template("landing.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/prompt1")
def prompt1():
    return render_template("prompt1.html")

@app.route("/prompt2")
def prompt2():
    return render_template("prompt2.html")

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/live_snapshot")
def live_snapshot():
    return render_template("live_snapshot.html")

@app.route("/capture_live_snapshot")
def capture_live_snapshot():
    try:
        image_path = capture_and_save_image()
        filename = os.path.basename(image_path)
        image_url = url_for("static", filename=f"captured_images/{filename}")
        return jsonify({"status": "success", "image_url": image_url})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/capturingimg")
def capture_img_with_loader():
    try:
        capture_and_save_image()
        return redirect(url_for('initialSnapshot'))
    except Exception as e:
        return f"Error capturing image: {str(e)}"

@app.route("/initialsnapshot")
def initialSnapshot():
    latest_image_path = get_latest_image()
    image_url = None
    if latest_image_path:
        filename = os.path.basename(latest_image_path)
        image_url = url_for('static', filename=f'captured_images/{filename}')
    return render_template("initial_snapshot.html", image_url=image_url)

@app.route("/predict", methods=['POST'])
def predict():
    latest = get_latest_image()
    if not latest:
        return jsonify({"status":"error","message":"No image found"})
    try:
        global size_matrix, predicted_sizes
        size_matrix, results, predicted_sizes = extract_size_matrix(latest)

        # Save YOLO bounding box visualization
        save_path = os.path.join(predicted_dir, "predicted_latest.jpg")
        results[0].save(filename=save_path)

        # Print tray grid in server logs
        print("=== Tray Size Matrix (5x6) ===")
        print(size_matrix)
        print("==============================")
        print(f"Detected sizes: {predicted_sizes}")

        image_url = url_for('static', filename='predicted_images/predicted_latest.jpg')
        return jsonify({
            "status":"success",
            "image_url":image_url,
            "grid": size_matrix.tolist(),
            "predicted_sizes": predicted_sizes
        })
    except Exception as e:
        return jsonify({"status":"error","message":str(e)})





@app.route('/actuate_size/<size>', methods=['POST'])
def actuate_size_route(size):
    global processed_sizes
    try:
        if size in processed_sizes:
            return jsonify({"status": "error", "message": f"Size {size.upper()} already processed."})

        actuate_size(size)   # run your servo actuation
        processed_sizes.add(size)  # mark as processed

        return jsonify({"status": "success", "message": f"Actuated size {size}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})





@app.route('/reset_moved', methods=['POST'])
def reset_moved_route():
    reset_moved_servos()
    return jsonify({"status":"success","message":"Reset moved servos"})

@app.route('/reset_all', methods=['POST'])
def reset_all_route():
    reset_all_servos()
    return jsonify({"status":"success","message":"Reset all servos"})


@app.route("/boundingboxSnapshot")
def boundingboxSnapshot():
    image_path = os.path.join(predicted_dir, 'predicted_latest.jpg')
    return render_template("boundingbox_snapshot.html", image_path=image_path)

@app.route("/sizeselect")
def selectSize():
    global processed_sizes, predicted_sizes
    show_invalid = request.args.get("show_invalid", "false") == "true"
    return render_template("size.html",
                           processed_sizes=list(processed_sizes),
                           predicted_sizes=list(predicted_sizes) if 'predicted_sizes' in globals() else [],
                           show_invalid=show_invalid)



@app.route("/slideout")
def slideOut():
    global processed_sizes, predicted_sizes
    # Use empty list if predicted_sizes doesn't exist yet
    predicted = list(predicted_sizes) if "predicted_sizes" in globals() else []
    available_sizes = [s for s in predicted if s not in processed_sizes]

    return render_template("slide_out.html",
                           available_sizes=available_sizes,
                           processed_sizes=list(processed_sizes),
                           predicted_sizes=predicted)



@app.route("/reset_processed", methods=["POST"])
def reset_processed():
    success, message = reset_processed_servos()
    if success:
        return jsonify({"status": "success", "message": message})
    else:
        return jsonify({"status": "error", "message": message})


@app.route("/slidein")
def slideIn():
    return render_template("slide_in.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/help")
def help():
    return render_template("help.html")

@app.route("/faq")
def faq():
    return render_template("faq.html")

@app.route('/shutdown', methods=['POST'])
def shutdown():
    if platform.system() == "Linux":
        try:
            subprocess.Popen(['sudo', 'shutdown', '-h', 'now'])
            return 'Raspberry Pi is shutting down...'
        except Exception as e:
            return f'Failed to shutdown: {str(e)}'
    else:
        return 'Shutdown command only works on Raspberry Pi/Linux.'

@app.route('/grid')
def get_grid():
    global size_matrix
    if size_matrix is None:
        return jsonify({"status":"error","message":"No grid available"})
    return jsonify({"status":"success","grid": size_matrix.tolist()})



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
