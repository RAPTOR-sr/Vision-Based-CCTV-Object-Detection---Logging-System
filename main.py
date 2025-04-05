# live_object_captioner/main.py
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import os
import numpy as np # Make sure numpy is imported

# Import our custom modules
from object_detector import ObjectDetector
from caption_generator import CaptionGenerator # <<< Updated import
from data_logger import DataLogger

# --- Configuration ---
VIDEO_SOURCE = 0  # 0 for default webcam, or use "path/to/video.mp4" or "rtsp://..."
YOLO_MODEL = 'yolov8n.pt' # Smaller YOLO model might be needed due to BLIP overhead
CONFIDENCE_THRESHOLD = 0.25 # Maybe slightly higher threshold
OUTPUT_DIR = 'output'
LOG_FILENAME = 'detections.csv'
LOG_INTERVAL_SECONDS = 3.0 # <<< New setting: Log every 3 seconds
# --- Configuration End ---

class Application:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.video_source = VIDEO_SOURCE
        self.vid = None
        self.is_running = False
        self.thread = None

        # --- Initialize Core Components ---
        print("Initializing components...")
        # 1. Object Detector
        self.detector = ObjectDetector(model_path=YOLO_MODEL)
        if self.detector.model is None:
             messagebox.showerror("Model Error", f"Failed to load YOLO model '{YOLO_MODEL}'. Please check the path and dependencies.")
             self.window.destroy()
             return

        # 2. Caption Generator (using BLIP)
        #    This might take a while the first time to download the model
        self.status_label_text = tk.StringVar(value="Status: Initializing BLIP model (may take time)...") # For status bar
        # Create UI elements that use status_label_text *before* potentially long init
        self._setup_ui_elements() # Separate UI setup
        self.window.update_idletasks() # Update UI to show the message
        self.captioner = CaptionGenerator() # <<< Uses BLIP now
        if self.captioner.model is None:
             messagebox.showerror("Model Error", "Failed to load BLIP captioning model. Check console for errors. Captioning disabled.")
             # Decide if you want to continue without captioning or exit
             # self.window.destroy()
             # return
        self.status_label_text.set("Status: Initializing Logger...") # Update status
        self.window.update_idletasks()

        # 3. Data Logger
        self.logger = DataLogger(log_dir=OUTPUT_DIR, filename=LOG_FILENAME)
        if self.logger.filepath is None:
             messagebox.showwarning("Logging Error", f"Failed to initialize CSV logger in '{OUTPUT_DIR}'. Logging will be disabled.")

        self.last_log_time = 0.0 # <<< Initialize log timer

        print("Initialization complete.")
        self.status_label_text.set("Status: Idle") # Final status

    def _setup_ui_elements(self):
        """Creates the UI widgets."""
        self.main_frame = ttk.Frame(self.window, padding="10")
        self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.main_frame, bg="lightgray")
        self.canvas.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        self.controls_frame = ttk.Frame(self.main_frame, padding="5")
        self.controls_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E))

        self.btn_start = ttk.Button(self.controls_frame, text="Start Feed", command=self.start_video)
        self.btn_start.grid(row=0, column=0, padx=5, pady=5)

        self.btn_stop = ttk.Button(self.controls_frame, text="Stop Feed", command=self.stop_video, state=tk.DISABLED)
        self.btn_stop.grid(row=0, column=1, padx=5, pady=5)

        # Use StringVar for status label to update it easily
        self.status_label = ttk.Label(self.main_frame, textvariable=self.status_label_text, relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(5,0))

        self.window.update_idletasks()
        self.window.minsize(640, 480)

    def start_video(self):
        if not self.is_running:
            self.status_label_text.set("Status: Initializing video source...")
            self.window.update_idletasks()

            try:
                self.vid = cv2.VideoCapture(self.video_source)
                if not self.vid.isOpened():
                    raise ValueError(f"Unable to open video source: {self.video_source}")

                width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if width > 0 and height > 0:
                    self.canvas.config(width=width, height=height)
                    # Consider a larger default window if captions are long
                    self.window.geometry(f"{max(width, 640)+40}x{height+120}")

                self.is_running = True
                self.last_log_time = time.time() # Reset log timer on start
                self.thread = threading.Thread(target=self.video_loop, daemon=True)
                self.thread.start()
                self.btn_start.config(state=tk.DISABLED)
                self.btn_stop.config(state=tk.NORMAL)
                self.status_label_text.set("Status: Running")
                print("Video feed started.")

            except Exception as e:
                 messagebox.showerror("Video Error", f"Failed to start video feed:\n{e}")
                 self.status_label_text.set("Status: Error starting video")
                 if self.vid:
                     self.vid.release()
                 self.vid = None

    def stop_video(self):
        if self.is_running:
            self.is_running = False
            if self.thread and self.thread.is_alive():
                 self.thread.join(timeout=2.0) # Give thread time to exit loop

            if self.vid:
                self.vid.release()
                self.vid = None

            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.status_label_text.set("Status: Stopped")
            self.canvas.delete("all")
            # Use winfo_width/height which should be available after geometry is set
            try:
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                self.canvas.create_text(canvas_width/2, canvas_height/2,
                                        text="Video Feed Stopped", fill="black", font=("Arial", 16))
            except tk.TclError: # Handle case where canvas might not be fully ready
                 print("Canvas not ready for text when stopping.")
            print("Video feed stopped.")


    def video_loop(self):
        frame_count = 0
        start_time = time.time()
        # last_log_time is now instance variable self.last_log_time

        while self.is_running and self.vid and self.vid.isOpened():
            ret, frame_bgr = self.vid.read()
            if not ret or frame_bgr is None:
                print("Failed to read frame from video source")
                time.sleep(0.1)
                continue

            try:
                # Add frame validation
                if frame_bgr.shape[0] == 0 or frame_bgr.shape[1] == 0:
                    print("Invalid frame dimensions")
                    continue
                    
                print("Processing frame...")
                # 1. Object Detection
                detections = self.detector.detect_objects(frame_bgr)
                print(f"Got {len(detections)} detections")

                current_time = time.time() # Get time at the start of processing the frame

                # --- Optional: Frame Skipping for Performance ---
                # process_this_frame = True # Or based on a counter/timer
                # if not process_this_frame:
                #     # Just display the frame without processing
                #     frame_rgb_display = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                #     self.window.after(0, self.update_canvas, frame_rgb_display)
                #     time.sleep(0.01) # Prevent busy loop
                #     continue # Skip rest of the loop

                # 1. Object Detection
                detections = self.detector.detect_objects(frame_bgr)

                # Prepare frame for drawing (do this once)
                frame_rgb_display = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                 # --- Check if it's time to log ---
                ready_to_log = (current_time - self.last_log_time >= LOG_INTERVAL_SECONDS)

                # 2. Process Detections, Generate Captions, Draw, and Log (conditionally)
                objects_logged_this_interval = 0
                for det in detections:
                    if det['confidence'] >= CONFIDENCE_THRESHOLD:
                        caption = "caption_pending" # Placeholder
                        try:
                            # 3. Generate Caption (Potentially slow!)
                            # Pass the original BGR frame for potential internal use
                            # by captioner, even if it converts to RGB/PIL later
                            caption = self.captioner.generate_caption(frame_bgr, det)

                            # 4. Log Data (ONLY if interval has passed)
                            if ready_to_log and self.logger.filepath:
                                self.logger.log_detection(det['class_name'], caption)
                                objects_logged_this_interval += 1

                        except Exception as cap_err:
                             print(f"Error during captioning or logging: {cap_err}")
                             caption = f"a {det['class_name']} (error)"


                        # 5. Draw Bounding Box and Caption on Frame
                        x1, y1, x2, y2 = det['bbox']
                        label = f"{det['class_name']}: {caption[:60]}{'...' if len(caption)>60 else ''}" # Show class + caption (truncated)
                        label += f" ({det['confidence']:.2f})"

                        # Basic text color logic (same as before)
                        text_color = (0, 0, 0) # Black
                        try:
                            roi_for_text = frame_rgb_display[max(0, y1-20):y1, x1:min(frame_rgb_display.shape[1], x1 + len(label)*8)]
                            if roi_for_text.size > 0 and np.mean(roi_for_text) < 128:
                                text_color = (255, 255, 255) # White
                        except Exception as e:
                            # print(f"Minor error checking text bg color: {e}")
                            pass

                        cv2.rectangle(frame_rgb_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame_rgb_display, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, text_color, 1, cv2.LINE_AA) # Smaller font

                # --- Reset log timer AFTER processing the frame if logging occurred ---
                if objects_logged_this_interval > 0:
                    self.last_log_time = current_time # Reset timer
                    print(f"Logged {objects_logged_this_interval} objects at {time.strftime('%H:%M:%S')}")


                # Calculate and Display FPS
                frame_count += 1
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(frame_rgb_display, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                # 6. Update Tkinter Canvas
                self.window.after(0, self.update_canvas, frame_rgb_display)

            except Exception as e:
                print(f"Error in video loop: {e}")
                import traceback
                traceback.print_exc() # Print detailed traceback
                # Optionally display error in status bar via main thread
                # self.window.after(0, self.status_label_text.set, f"Status: Error - {e}")
                time.sleep(0.1)

        print("Video loop finished.")

    # update_canvas function remains the same as before

    def update_canvas(self, frame_rgb):
        """Updates the Tkinter canvas with the new frame. Must be called from the main thread."""
        if self.is_running and self.canvas.winfo_exists(): # Check if canvas still exists
            try:
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                frame_height, frame_width = frame_rgb.shape[:2]

                if canvas_width > 1 and canvas_height > 1 and frame_width > 0 and frame_height > 0:
                    scale = min(canvas_width / frame_width, canvas_height / frame_height)
                    new_w = int(frame_width * scale)
                    new_h = int(frame_height * scale)

                    resized_frame = cv2.resize(frame_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    self.photo = ImageTk.PhotoImage(image=Image.fromarray(resized_frame))
                    self.canvas.delete("all")
                    x_offset = (canvas_width - new_w) // 2
                    y_offset = (canvas_height - new_h) // 2
                    self.canvas.create_image(x_offset, y_offset, image=self.photo, anchor=tk.NW)
                else:
                     self.canvas.delete("all")
                     self.canvas.create_text(self.canvas.winfo_width()/2, self.canvas.winfo_height()/2, text="Waiting for valid frame/canvas size...", fill="black")

            except Exception as e:
                print(f"Error updating canvas: {e}")


    # on_closing function remains the same as before
    def on_closing(self):
        """Called when the user closes the window."""
        print("Close button pressed.")
        self.stop_video() # Ensure video stops cleanly
        self.window.destroy() # Destroy the Tkinter window


# --- Main Execution ---
if __name__ == "__main__":
    print("Starting application...")
    if not os.path.exists(OUTPUT_DIR):
        try:
            os.makedirs(OUTPUT_DIR)
            print(f"Created output directory: {OUTPUT_DIR}")
        except OSError as e:
            messagebox.showerror("Directory Error", f"Could not create output directory '{OUTPUT_DIR}':\n{e}")

    root = tk.Tk()
    app = Application(root, "Live Object Detection and BLIP Captioning")
    root.mainloop()
    print("Application finished.")