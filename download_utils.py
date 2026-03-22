"""
Download utility module for saving transcriptions to file without page resets.
"""
import os
import streamlit as st


def save_transcript(text, model_display_name, filename):
    """
    Save transcription to file in output directory.
    
    Args:
        text: The transcription text to save
        model_display_name: Display name of the model (e.g., "Whisper Small (Finetuned-Quran)")
        filename: Original uploaded filename (without extension)
    
    Returns:
        output_path: Full path to the saved file
    """
    os.makedirs("output", exist_ok=True)
    filename_clean = filename.rsplit('.', 1)[0]
    output_path = f"output/{model_display_name.replace(' ', '_')}-{filename_clean}.txt"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    return output_path


def show_download_button(text, model_display_name, filename):
    """
    Display download button and handle saving without page resets.
    Uses session state to avoid re-rendering issues.
    
    Args:
        text: The transcription text
        model_display_name: Display name of the model
        filename: Original uploaded filename
    """
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if st.button("📥 Download Transcript", key="download_btn_main"):
            try:
                output_path = save_transcript(text, model_display_name, filename)
                st.session_state.last_download_path = output_path
                st.session_state.download_success = True
            except Exception as e:
                st.session_state.download_error = str(e)
                st.session_state.download_success = False
    
    # Display success or error message without resets
    if "download_success" in st.session_state:
        if st.session_state.download_success:
            st.success(f"✅ Saved to: {st.session_state.last_download_path}")
        else:
            st.error(f"❌ Error saving file: {st.session_state.download_error}")
