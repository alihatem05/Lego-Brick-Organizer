
import cv2
import numpy as np
import joblib
import os
import time
from pathlib import Path

from configs import MODEL_OUT_DIR, IMAGE_SIZE
from features import extract_all_features
from preprocessing import resize, normalize
from utils import ensure_dir

class LegoDetector:
    
    def __init__(self, model_name='Random Forest'):
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.class_names = None
        self.feature_selector = None
        self.load_model()
    
    def load_model(self):
        print(f"Loading model: {self.model_name}")
        
        model_file = f"{self.model_name.lower().replace(' ', '_')}.pkl"
        model_path = os.path.join(MODEL_OUT_DIR, model_file)
        
        if not os.path.exists(model_path):
            available_models = [f.replace('.pkl', '').replace('_', ' ').title() 
                              for f in os.listdir(MODEL_OUT_DIR) 
                              if f.endswith('.pkl') and f not in ['scaler.pkl', 'class_names.pkl', 'feature_selector.pkl']]
            raise FileNotFoundError(
                f"Model '{self.model_name}' not found at {model_path}\n"
                f"Available models: {available_models}"
            )
        
        self.model = joblib.load(model_path)
        print(f"✓ Loaded model: {self.model_name}")
        
        scaler_path = os.path.join(MODEL_OUT_DIR, "scaler.pkl")
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print("✓ Loaded feature scaler")
        
        classes_path = os.path.join(MODEL_OUT_DIR, "class_names.pkl")
        if os.path.exists(classes_path):
            self.class_names = joblib.load(classes_path)
            print(f"✓ Loaded {len(self.class_names)} classes: {self.class_names}")
        
        selector_path = os.path.join(MODEL_OUT_DIR, "feature_selector.pkl")
        if os.path.exists(selector_path):
            self.feature_selector = joblib.load(selector_path)
            print("✓ Loaded feature selector")
        
        print("Model loaded successfully!\n")
    
    def preprocess_frame(self, frame):
        frame_resized = resize(frame, IMAGE_SIZE)
        
        frame_norm = normalize(frame_resized)
        
        frame_uint8 = (frame_norm * 255).astype(np.uint8)
        
        return frame_uint8
    
    def predict(self, frame):
        processed = self.preprocess_frame(frame)
        
        features = extract_all_features(processed)
        features = features.reshape(1, -1)
        
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        if self.feature_selector is not None:
            features = features[:, self.feature_selector]
        
        prediction = self.model.predict(features)[0]
        
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities[prediction]
        else:
            probabilities = None
            confidence = 1.0
        
        class_name = self.class_names[prediction] if self.class_names else str(prediction)
        
        return class_name, confidence, probabilities
    
    def draw_prediction(self, frame, class_name, confidence, fps=None):
        height, width = frame.shape[:2]
        
        overlay = frame.copy()
        
        cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        text = f"Predicted: {class_name}"
        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.2, (0, 255, 0), 2, cv2.LINE_AA)
        
        conf_text = f"Confidence: {confidence*100:.1f}%"
        cv2.putText(frame, conf_text, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        if fps is not None:
            fps_text = f"FPS: {fps:.1f}"
            cv2.putText(frame, fps_text, (width - 150, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        
        instructions = "Press 'q' to quit, 'c' to capture, 's' to switch model"
        cv2.putText(frame, instructions, (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        
        return frame
    
    def run_detection(self, camera_id=0, show_fps=True):
        print("\n" + "="*70)
        print("STARTING REAL-TIME LEGO BRICK DETECTION")
        print("="*70)
        print(f"Using model: {self.model_name}")
        print(f"Camera ID: {camera_id}")
        print("\nControls:")
        print("  'q' - Quit")
        print("  'c' - Capture and save current frame")
        print("  's' - Show detection statistics")
        print("  SPACE - Pause/Resume")
        print("="*70 + "\n")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_id}")
            print("Try using a different camera_id (e.g., 0, 1, 2...)")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        fps = 0
        frame_count = 0
        start_time = time.time()
        
        predictions_history = []
        capture_count = 0
        
        paused = False
        
        print("Camera opened successfully! Starting detection...")
        
        try:
            while True:
                if not paused:
                    ret, frame = cap.read()
                    
                    if not ret:
                        print("Error: Could not read frame")
                        break
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    class_name, confidence, probabilities = self.predict(frame_rgb)
                    predictions_history.append((class_name, confidence))
                    
                    frame_count += 1
                    if frame_count % 10 == 0:
                        elapsed_time = time.time() - start_time
                        fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    display_frame = self.draw_prediction(
                        frame, class_name, confidence, fps if show_fps else None
                    )
                else:
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (0, 200), (640, 280), (0, 0, 0), -1)
                    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
                    cv2.putText(frame, "PAUSED", (250, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3, cv2.LINE_AA)
                    display_frame = frame
                
                cv2.imshow('LEGO Brick Detector', display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('c'):
                    capture_dir = "captures"
                    ensure_dir(capture_dir)
                    capture_path = os.path.join(capture_dir, f"capture_{capture_count:04d}.jpg")
                    cv2.imwrite(capture_path, frame)
                    capture_count += 1
                    print(f"Captured: {capture_path}")
                elif key == ord('s'):
                    print("\n" + "-"*50)
                    print("DETECTION STATISTICS")
                    print("-"*50)
                    if predictions_history:
                        unique_classes = {}
                        for cls, conf in predictions_history[-100:]:
                            unique_classes[cls] = unique_classes.get(cls, 0) + 1
                        print("Recent predictions:")
                        for cls, count in sorted(unique_classes.items(), key=lambda x: x[1], reverse=True):
                            print(f"  {cls}: {count}")
                    print(f"Average FPS: {fps:.2f}")
                    print(f"Total frames: {frame_count}")
                    print(f"Captures saved: {capture_count}")
                    print("-"*50 + "\n")
                elif key == ord(' '):
                    paused = not paused
                    print("PAUSED" if paused else "RESUMED")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            print("\n" + "="*70)
            print("DETECTION SESSION SUMMARY")
            print("="*70)
            print(f"Total frames processed: {frame_count}")
            print(f"Average FPS: {fps:.2f}")
            print(f"Captures saved: {capture_count}")
            print("="*70)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time LEGO Brick Detection')
    parser.add_argument('--model', type=str, default='Random Forest',
                       help='Model to use (e.g., "Random Forest", "SVM", "XGBoost")')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--no-fps', action='store_true',
                       help='Hide FPS display')
    
    args = parser.parse_args()
    
    try:
        detector = LegoDetector(model_name=args.model)
        
        detector.run_detection(
            camera_id=args.camera,
            show_fps=not args.no_fps
        )
    
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("\nPlease train models first by running:")
        print("  python train.py")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()