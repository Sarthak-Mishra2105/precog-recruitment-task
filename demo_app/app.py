"""
Colored MNIST Demo App
======================
Interactive demo where users can draw digits and see predictions from:
- Cheater (color-biased) model
- Consistency (debiased) model  
- GRL (debiased) model

Plus Grad-CAM visualizations for each model.

Run with: streamlit run demo_app/app.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Import from project
from models import SimpleCNN
from debias import GRLModel
from interpretability import GradCAM
from preprocess import (
    COLORS, COLOR_NAMES, 
    canvas_to_tensor, recolor_tensor
)


# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Colored MNIST Demo",
    page_icon="🎨",
    layout="wide"
)


# ============================================================================
# MODEL LOADING (cached)
# ============================================================================
@st.cache_resource
def load_models():
    """Load all three models from artifacts."""
    device = torch.device('cpu')
    artifacts = PROJECT_ROOT / "artifacts"
    
    models = {}
    
    # Cheater baseline
    cheater = SimpleCNN(num_classes=10).to(device)
    ckpt = torch.load(artifacts / "baseline_cheater_N2000.pt", 
                      map_location=device, weights_only=False)
    cheater.load_state_dict(ckpt['model_state_dict'])
    cheater.eval()
    models['Cheater'] = cheater
    
    # Consistency model
    consistency = SimpleCNN(num_classes=10).to(device)
    ckpt = torch.load(artifacts / "model_consistency.pt", 
                      map_location=device, weights_only=False)
    consistency.load_state_dict(ckpt['model_state_dict'])
    consistency.eval()
    models['Consistency'] = consistency
    
    # GRL model
    grl = GRLModel(lambda_grl=1.0).to(device)
    ckpt = torch.load(artifacts / "model_grl.pt", 
                      map_location=device, weights_only=False)
    grl.load_state_dict(ckpt['model_state_dict'])
    grl.eval()
    models['GRL'] = grl
    
    return models, device


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================
def get_prediction(model, tensor, is_grl=False):
    """Get prediction with confidence and top-3."""
    with torch.no_grad():
        if is_grl:
            logits = model.predict(tensor)
        else:
            logits = model(tensor)
        
        probs = F.softmax(logits, dim=1)[0]
        pred = probs.argmax().item()
        conf = probs[pred].item()
        
        # Top-3
        top3_vals, top3_idx = probs.topk(3)
        top3 = [(idx.item(), val.item()) for idx, val in zip(top3_idx, top3_vals)]
        
    return pred, conf, top3


def compute_gradcam(model, tensor, target_class=None, is_grl=False):
    """Compute Grad-CAM for a model."""
    try:
        if is_grl:
            backbone = model.backbone
            target_layer = 'conv3.0'
        else:
            backbone = model
            target_layer = 'conv3.0'
        
        gradcam = GradCAM(backbone, target_layer)
        cam = gradcam(tensor.clone().requires_grad_(True), target_class)
        cam_up = gradcam.upsample(cam, (28, 28))
        
        # Create overlay
        img_np = tensor.squeeze(0).permute(1, 2, 0).numpy()
        overlay = gradcam.overlay(img_np, cam_up)
        
        return cam_up.numpy(), overlay
    except Exception as e:
        st.warning(f"Grad-CAM error: {e}")
        return np.zeros((28, 28)), tensor.squeeze(0).permute(1, 2, 0).numpy()


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_model_card(model_name, pred, conf, top3, is_correct=None):
    """Render a model prediction card."""
    # Color based on correctness
    if is_correct is True:
        border = "2px solid #28a745"
    elif is_correct is False:
        border = "2px solid #dc3545"
    else:
        border = "1px solid #ddd"
    
    st.markdown(f"""
    <div style="border: {border}; border-radius: 10px; padding: 15px; text-align: center;">
        <h3 style="margin: 0;">{model_name}</h3>
        <h1 style="font-size: 48px; margin: 10px 0; color: #4A90A4;">{pred}</h1>
        <p style="font-size: 20px; margin: 0;">Confidence: {conf:.1%}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Top-3 table
    st.markdown("**Top-3 Predictions:**")
    for digit, prob in top3:
        bar_width = int(prob * 100)
        st.markdown(f"""
        <div style="display: flex; align-items: center; margin: 2px 0;">
            <span style="width: 20px;">{digit}</span>
            <div style="flex: 1; background: #eee; border-radius: 3px; margin: 0 5px;">
                <div style="width: {bar_width}%; background: #4A90A4; height: 8px; border-radius: 3px;"></div>
            </div>
            <span style="width: 40px; font-size: 12px;">{prob:.1%}</span>
        </div>
        """, unsafe_allow_html=True)


def render_gradcam_image(image, overlay, cam, title):
    """Render Grad-CAM visualization."""
    fig, axes = plt.subplots(1, 3, figsize=(6, 2))
    
    axes[0].imshow(np.clip(image, 0, 1))
    axes[0].set_title("Input", fontsize=8)
    axes[0].axis('off')
    
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title("CAM", fontsize=8)
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay", fontsize=8)
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=9)
    plt.tight_layout()
    return fig


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.title("🎨 Colored MNIST Demo")
    st.markdown("""
    Draw a digit and see how different models respond to color!
    - **Cheater**: Learns color shortcut (predicts based on color, not shape)
    - **Consistency**: Debiased via color consistency training
    - **GRL**: Debiased via gradient reversal
    """)
    
    # Load models
    try:
        models, device = load_models()
        st.success("✅ Models loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.stop()
    
    # Sidebar controls
    st.sidebar.header("Controls")
    
    color_id = st.sidebar.selectbox(
        "Stroke Color",
        range(10),
        format_func=lambda x: f"C{x}: {COLOR_NAMES[x]}"
    )
    
    # Show color preview
    color_rgb = COLORS[color_id]
    color_hex = '#{:02x}{:02x}{:02x}'.format(
        int(color_rgb[0]*255), int(color_rgb[1]*255), int(color_rgb[2]*255)
    )
    st.sidebar.markdown(f"""
    <div style="background: {color_hex}; width: 100%; height: 30px; border-radius: 5px;"></div>
    """, unsafe_allow_html=True)
    
    brush_size = st.sidebar.slider("Brush Size", 5, 30, 15)
    
    # Main layout
    col_canvas, col_results = st.columns([1, 2])
    
    with col_canvas:
        st.subheader("Draw a Digit")
        
        # Canvas
        canvas_result = st_canvas(
            fill_color="rgba(0, 0, 0, 0)",
            stroke_width=brush_size,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        
        st.caption("Draw on the canvas above (white strokes on black)")
    
    # Check if there's a drawing
    has_drawing = (canvas_result.image_data is not None and 
                   canvas_result.image_data.max() > 0)
    
    with col_results:
        if has_drawing:
            # Convert canvas to tensor
            tensor = canvas_to_tensor(canvas_result.image_data, color_id)
            
            # Show preprocessed image
            st.subheader("Preprocessed Input")
            img_np = tensor.squeeze(0).permute(1, 2, 0).numpy()
            
            # Scale up for display
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil = img_pil.resize((140, 140), Image.Resampling.NEAREST)
            st.image(img_pil, caption=f"28×28 input (Color: {COLOR_NAMES[color_id]})")
            
            # Model predictions
            st.subheader("Model Predictions")
            pred_cols = st.columns(3)
            
            predictions = {}
            for idx, (name, model) in enumerate(models.items()):
                is_grl = (name == 'GRL')
                pred, conf, top3 = get_prediction(model, tensor, is_grl)
                predictions[name] = (pred, conf, top3)
                
                with pred_cols[idx]:
                    render_model_card(name, pred, conf, top3)
            
            # Grad-CAM
            st.subheader("Grad-CAM Visualizations")
            cam_cols = st.columns(3)
            
            for idx, (name, model) in enumerate(models.items()):
                is_grl = (name == 'GRL')
                pred = predictions[name][0]
                cam, overlay = compute_gradcam(model, tensor, pred, is_grl)
                
                with cam_cols[idx]:
                    fig = render_gradcam_image(img_np, overlay, cam, f"{name}: class {pred}")
                    st.pyplot(fig, width="stretch")
                    plt.close(fig)
        else:
            st.info("👆 Draw a digit on the canvas to see predictions!")
    
    # Recolor test section
    st.markdown("---")
    st.subheader("🔄 Recolor Test")
    st.markdown("See how predictions change when the same digit is shown in different colors:")
    
    if has_drawing:
        # Create recolor test table
        tensor = canvas_to_tensor(canvas_result.image_data, 0)  # Start with any color
        
        # Table data
        table_data = []
        for cid in range(10):
            recolored = recolor_tensor(tensor, cid)
            row = {'Color': f"C{cid}: {COLOR_NAMES[cid]}"}
            
            for name, model in models.items():
                is_grl = (name == 'GRL')
                pred, conf, _ = get_prediction(model, recolored, is_grl)
                row[name] = f"{pred} ({conf:.0%})"
            
            table_data.append(row)
        
        # Display as table
        import pandas as pd
        df = pd.DataFrame(table_data)
        st.dataframe(df, hide_index=True, width="stretch")
        
        # Visual strip
        st.markdown("**Visual prediction strip:**")
        strip_cols = st.columns(10)
        
        for cid in range(10):
            recolored = recolor_tensor(tensor, cid)
            
            with strip_cols[cid]:
                # Get predictions
                preds = []
                for name, model in models.items():
                    is_grl = (name == 'GRL')
                    pred, _, _ = get_prediction(model, recolored, is_grl)
                    preds.append(str(pred))
                
                # Show image and predictions
                img = recolored.squeeze(0).permute(1, 2, 0).numpy()
                st.image(img, caption=f"C{cid}", width="stretch")
                st.caption("/".join(preds))
    else:
        st.info("Draw a digit first to see the recolor test.")


if __name__ == "__main__":
    main()
