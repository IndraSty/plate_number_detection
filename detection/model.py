# Import libraries
from ultralytics import YOLO
import easyocr

def initialize_models(st):
    """Initialize YOLO and EasyOCR models"""
    try:

        if 'model' not in st.session_state:
            with st.spinner("🔄 Loading YOLO model..."):
                st.session_state.model = YOLO('yolov8n.pt') 
        
        if 'reader' not in st.session_state:
            with st.spinner("🔄 Loading OCR model..."):
                st.session_state.reader = easyocr.Reader(['en'], gpu=False)
        
        return st.session_state.model, st.session_state.reader
    
    except ImportError as e:
        st.error(f"❌ Missing required libraries: {e}")
        st.info("Please install: pip install ultralytics easyocr opencv-python")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None