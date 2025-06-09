import streamlit as st
import cv2
import pandas as pd
import numpy as np
import tempfile
import os

from detection.model import initialize_models
from detection import process_video


# Setup page config
st.set_page_config(
    page_title="🚗 License Plate Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🚗 License Plate Detection System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        confidence_threshold = st.slider(
            "Vehicle Detection Confidence", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.5, 
            step=0.1,
            help="Higher values = fewer but more confident detections"
        )
        
        process_frames = st.selectbox(
            "Process Every N Frames",
            options=[1, 3, 5, 10, 15],
            index=2,  # Default to 5
            help="Higher values = faster processing, lower accuracy"
        )
        
        st.markdown("---")
        
        st.markdown("""
        ### 📋 Instructions:
        1. Upload a video file
        2. Adjust settings if needed
        3. Click 'Process Video'
        4. View results and download
        
        ### 🎯 Supported Formats:
        - MP4, AVI, MOV, MKV
        - Max size: 200MB
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Video")
        
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload your video file for license plate detection"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_path = tmp_file.name
            
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
            
            # Display video info
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                
                st.markdown(f"""
                <div class="info-box">
                <h4>📹 Video Information:</h4>
                <ul>
                <li><strong>Duration:</strong> {duration:.1f} seconds</li>
                <li><strong>FPS:</strong> {fps}</li>
                <li><strong>Resolution:</strong> {width}x{height}</li>
                <li><strong>Total Frames:</strong> {frame_count}</li>
                </ul>
                </div>
                """, unsafe_allow_html=True)
            
            # Show original video
            st.video(uploaded_file)
            
            # Process button
            if st.button("🚀 Process Video", type="primary"):
                # Initialize models
                model, reader = initialize_models(st)
                
                if model is not None and reader is not None:
                    st.header("🔄 Processing...")
                    
                    # Process video
                    output_path, detections, stats = process_video(
                        st,
                        video_path, model, reader, 
                        confidence_threshold, process_frames
                    )
                    
                    if output_path and detections:
                        # Store results in session state
                        st.session_state.output_path = output_path
                        st.session_state.detections = detections
                        st.session_state.stats = stats
                        st.session_state.original_filename = uploaded_file.name
                        
                        st.success("✅ Processing completed successfully!")
                        st.rerun()  # Refresh to show results
    
    with col2:
        st.header("📊 Results")
        
        # Show results if available
        if hasattr(st.session_state, 'output_path') and st.session_state.output_path:
            
            # Display processed video
            st.subheader("🎬 Processed Video")
            
            if os.path.exists(st.session_state.output_path):
                with open(st.session_state.output_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                st.video(video_bytes)
                
                # Download button for processed video
                st.download_button(
                    label="📥 Download Processed Video",
                    data=video_bytes,
                    file_name=f"processed_{st.session_state.original_filename}",
                    mime="video/mp4"
                )
            
            # Statistics
            st.subheader("📈 Statistics")
            stats = st.session_state.stats
            
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.metric("Total Detections", stats['total_detections'])
                st.metric("Processing Time", f"{stats['processing_time']:.1f}s")
            
            with col_stat2:
                st.metric("Unique Plates", stats['unique_plates'])
                st.metric("Video Duration", f"{stats['duration']:.1f}s")
            
            with col_stat3:
                st.metric("Frames Processed", stats['processed_frames'])
                st.metric("Video FPS", stats['fps'])
            
            # Vehicle types chart
            if stats['vehicle_types']:
                st.subheader("🚙 Vehicle Types Detected")
                vehicle_df = pd.DataFrame(
                    list(stats['vehicle_types'].items()),
                    columns=['Vehicle Type', 'Count']
                )
                st.bar_chart(vehicle_df.set_index('Vehicle Type'))
            
            # Detection details
            st.subheader("🔍 Detection Details")
            
            if st.session_state.detections:
                detections_df = pd.DataFrame(st.session_state.detections)
                
                # Filter options
                vehicle_filter = st.multiselect(
                    "Filter by Vehicle Type:",
                    options=detections_df['vehicle_type'].unique(),
                    default=detections_df['vehicle_type'].unique()
                )
                
                plate_only = st.checkbox("Show only detections with plates")
                
                # Apply filters
                filtered_df = detections_df[detections_df['vehicle_type'].isin(vehicle_filter)]
                if plate_only:
                    filtered_df = filtered_df[filtered_df['plate_text'] != "Not detected"]
                
                # Display table
                st.dataframe(
                    filtered_df[['timestamp', 'vehicle_type', 'plate_text', 'vehicle_confidence', 'plate_confidence']].round(2),
                    use_container_width=True
                )
                
                # Download CSV
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Detection Data (CSV)",
                    data=csv,
                    file_name=f"detections_{st.session_state.original_filename.replace('.mp4', '.csv')}",
                    mime="text/csv"
                )
                
                # License plates summary
                unique_plates = filtered_df[filtered_df['plate_text'] != "Not detected"]['plate_text'].unique()
                if len(unique_plates) > 0:
                    st.subheader("🔢 Detected License Plates")
                    for i, plate in enumerate(unique_plates, 1):
                        st.write(f"{i}. **{plate}**")
            
            else:
                st.info("No detections found in the video.")
        
        else:
            st.info("👆 Upload a video and click 'Process Video' to see results here.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    🚗 License Plate Detection System | Built with Streamlit, YOLOv8, and EasyOCR
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()