import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from io import BytesIO
from dataclasses import dataclass
import sklearn
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

# ----------------- Page Config ------------------
st.set_page_config(
    page_title="🧊 Glacier Melting Tracker | Portfolio",
    layout="wide",
    page_icon="🧊"
)

# ----------------- Theme / Styles ------------------
PRIMARY = "#00B5D8"  # teal-ish
ACCENT = "#A855F7"  # purple
BG = "#0B1220"
CARD_BG = "#111827"
TEXT = "#E5E7EB"
SUBTEXT = "#9CA3AF"

st.markdown(
    f"""
    <style>
    .block-container {{
        padding-top: 2rem; padding-bottom: 2rem; max-width: 1300px;
    }}
    html, body, [class*="css"]  {{
        color: {TEXT}; background-color: {BG};
    }}
    .big-title {{
        font-size: 42px; font-weight: 800; letter-spacing: 0.2px; margin-bottom: 0.25rem;
    }}
    .subtitle {{ color: {SUBTEXT}; margin-top: 0rem; }}
    .card {{
        background: {CARD_BG}; padding: 1.25rem 1.4rem; border-radius: 1.2rem; border: 1px solid #1f2937;
        box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    }}
    .pill {{ display:inline-block; padding: 0.25rem 0.6rem; border-radius: 999px; border:1px solid #374151; color:{SUBTEXT}; font-size: 0.8rem; margin-right: 0.4rem; }}
    .btn-primary {{ background:{PRIMARY}; color:#001018; padding:0.65rem 1rem; border-radius:0.8rem; text-decoration:none; font-weight:700; }}
    .metric-kpi {{ display:flex; gap:0.5rem; align-items:center; }}
    .metric-kpi .value {{ font-size: 28px; font-weight: 800; color: {TEXT}; }}
    .metric-kpi .label {{ color: {SUBTEXT}; font-size: 12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- Routing ------------------
PAGES = ["Home", "Glacier Tracker", "Portfolio", "Team", "About"]
choice = st.sidebar.selectbox("Navigate", PAGES)


# ----------------- Utilities ------------------
@dataclass
class SegmentParams:
    # HSV thresholds for ice/snow (bright, low-saturation)
    h_min: int = 0
    s_max: int = 60
    v_min: int = 150


def segment_ice_hsv(img_bgr: np.ndarray, params: SegmentParams) -> np.ndarray:
    """Return binary mask where ice/snow detected using HSV heuristics."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # White/ice tends to have low saturation and high value
    lower = np.array([params.h_min, 0, params.v_min], dtype=np.uint8)
    upper = np.array([179, params.s_max, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # Clean up: remove noise and fill holes
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def segment_ice_kmeans(img_bgr: np.ndarray, k: int = 3) -> np.ndarray:
    """KMeans on color to isolate the brightest cluster as ice; returns binary mask."""
    Z = img_bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _ret, labels, centers = cv2.kmeans(Z, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(centers)
    # Assume brightest cluster (in grayscale) is glaciers/ice/snow
    brightness = centers.mean(axis=1)
    ice_cluster = np.argmax(brightness)
    mask = (labels.flatten() == ice_cluster).astype(np.uint8) * 255
    mask = mask.reshape(img_bgr.shape[:2])
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def area_ratio(mask: np.ndarray) -> float:
    """Percentage of image considered ice."""
    white = (mask > 0).sum()
    total = mask.size
    return (white / total) * 100.0 if total else 0.0


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_image(file) -> np.ndarray:
    bytes_data = file.read()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    return img


def annotate_mask(img_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = img_bgr.copy()
    color = np.zeros_like(img_bgr)
    color[:, :] = (255, 0, 0)  # Blue overlay in BGR
    alpha = 0.35
    overlay = cv2.addWeighted(overlay, 1.0, color, 0.0, 0.0)
    overlay[mask > 0] = cv2.addWeighted(overlay[mask > 0], 1 - alpha, color[mask > 0], alpha, 0)
    # Draw mask edges
    edges = cv2.Canny(mask, 50, 150)
    overlay[edges > 0] = (0, 255, 255)  # yellow edges
    return overlay


def process_video(file, method: str, params: SegmentParams, sample_every: int = 10, max_frames: int = 400):
    bytes_data = file.read()
    arr = np.frombuffer(bytes_data, np.uint8)
    vid = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    # Fallback if imdecode fails for video; use VideoCapture on temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(bytes_data)
        path = tmp.name
    cap = cv2.VideoCapture(path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    records = []
    frames_preview = []
    idx = 0
    sampled = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            if method == "HSV":
                mask = segment_ice_hsv(frame, params)
            else:
                mask = segment_ice_kmeans(frame)
            ratio = area_ratio(mask)
            t = idx / fps
            records.append({"frame": idx, "time_s": round(t, 2), "ice_%": ratio})
            if len(frames_preview) < 6:
                frames_preview.append((to_rgb(frame), mask, to_rgb(annotate_mask(frame, mask))))
            sampled += 1
            if sampled >= max_frames:
                break
        idx += 1
    cap.release()

    df = pd.DataFrame(records)
    return df, frames_preview


# ----------------- Pages ------------------
if choice == "Home":
    st.markdown(f"<div class='big-title'>🧊 Glacier Melting <span style='color:{PRIMARY}'>Tracker</span></div>",
                unsafe_allow_html=True)

    st.markdown(
        f"<p class='subtitle'>Computer vision demo for estimating glacier/snow area over time from images or video. Upload two images (before/after) or a video; tune thresholds; get metrics & interactive graphs.</p>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            "<div class='card'>\n<div class='metric-kpi'><span class='value'>2</span><span class='label'>Modes</span></div>\n<p class='subtitle'>Image pair & Video time-series</p></div>",
            unsafe_allow_html=True)
    with c2:
        st.markdown(
            "<div class='card'>\n<div class='metric-kpi'><span class='value'>Interactive</span></div>\n<p class='subtitle'>Adjust HSV or KMeans; live masks</p></div>",
            unsafe_allow_html=True)
    with c3:
        st.markdown(
            "<div class='card'>\n<div class='metric-kpi'><span class='value'>Exports</span></div>\n<p class='subtitle'>Download CSV & figures</p></div>",
            unsafe_allow_html=True)

    st.markdown("""
    **Quick Start**
    1. Go to **Glacier Tracker**.
    2. Select **Image Pair** or **Video**.
    3. Upload files, choose method (**HSV** or **KMeans**), tweak sliders.
    4. Review **ice coverage metrics** and **graphs**.
    5. **Download** results.
    """)
    st.markdown("## 📌 Portfolio: Computer Vision for Tracking Glacier Melting Using Python")

    st.write(
        "This portfolio showcases a Python-based project leveraging computer vision techniques "
        "to monitor and track glacier melting. The core model simulates ICEpy4D concepts using "
        "available libraries for demonstration purposes."
    )

    # --- Home Page Overview ---
    with st.expander("🏠 Home Page Overview", expanded=True):
        st.write("Welcome to the Glacier Monitoring Portfolio!")
        st.write(
            "This project demonstrates how Python and computer vision can combat climate change "
            "by tracking glacier melting in real-time. Using affordable hardware and advanced algorithms, "
            "we enable scientists, researchers, and environmentalists to monitor remote glaciers "
            "without expensive equipment."
        )
        st.markdown("**Key Highlights:**")
        st.markdown("""
           - 📡 **Real-Time Tracking**: Analyze image sequences to detect melt rates  
           - 📊 **Data-Driven Insights**: Generate visualizations for trends and predictions  
           - 🔓 **Open-Source Inspiration**: Based on ICEpy4D for easy extension  
           """)

    # --- Team Details ---
    with st.expander("👨‍🔬 Team Details"):
        st.markdown("""
           | Name              | Role/Affiliation                                   | Contribution |
           |-------------------|----------------------------------------------------|--------------|
           | **Francesco Ioli** | Lead Developer / PhD Researcher, Politecnico di Milano | Core package development, deep learning integration, and photogrammetry tools |
           | **Nicolò Dematteis** | Researcher, Italian National Research Council (CNR) | Glacier monitoring expertise and data validation |
           | **Daniele Giordan** | Researcher, CNR | Field deployment and stereo-camera setup |
           | **Francesco Nex** | Professor, University of Twente | SfM algorithms and computer vision guidance |
           | **Livio Pinto** | Collaborator, CNR | Testing and multi-epoch processing |
           """)

    # --- Features ---
    with st.expander("✨ Attractive Features"):
        st.markdown("""
           - 🤖 **Deep Learning Simulation**: Torch for basic feature detection simulation  
           - 💻 **Low-Cost Setup Simulation**: Synthetic images for 4D reconstruction demo  
           - 🕒 **Multi-Epoch Processing**: Handles time series data for melt tracking  
           - 📈 **Visualization Tools**: Matplotlib for graphs and maps  
           - 📊 **Data Analysis**: Pandas for glacier data trends  
           - 🧊 **Change Detection**: Image differencing to estimate melt rates  
           - 🧩 **Extensibility**: Modular functions for adding custom CV components  
           - ⚡ **CUDA Support Check**: Detects GPU availability in Torch  
           - 📉 **Interactive Plots**: Dynamic visualizations with Streamlit  
           - 🔬 **Scientific Reproducibility**: Sample data + clear code structure  
           """)
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd


    def glacier_mapping_graph():
        # Simulated time series (days)
        days = np.arange(1, 11)  # 10 days
        # Simulated melt volume (pixel change or cubic meters)
        melt_volume = np.random.uniform(5, 15, size=len(days)).cumsum()

        df = pd.DataFrame({"Day": days, "Melt Volume": melt_volume})

        st.markdown("## 🧊 Glacier Mapping & Melting Trend")
        st.line_chart(df.set_index("Day"))

        # Matplotlib version (for more control)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(days, melt_volume, marker="o", linestyle="-", linewidth=2)
        ax.set_title("Glacier Melt Volume Tracking", fontsize=14)
        ax.set_xlabel("Day")
        ax.set_ylabel("Estimated Melt Volume (arbitrary units)")
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)


    # Call inside your Streamlit app
    glacier_mapping_graph()
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression


    def glacier_mapping_regression_classical():
        st.markdown("## 🧊 Glacier Mapping Graph (Regression vs Classical)")

        # --- Simulated classical glacier data ---
        days = np.arange(1, 11).reshape(-1, 1)  # Days 1–10
        classical_melt = np.array([5, 7, 9, 12, 15, 18, 22, 26, 29, 35])  # Example melt values

        # --- Regression model ---
        model = LinearRegression()
        model.fit(days, classical_melt)
        regression_melt = model.predict(days)

        # --- Future prediction (next 5 days) ---
        future_days = np.arange(11, 16).reshape(-1, 1)
        future_pred = model.predict(future_days)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(days, classical_melt, color="blue", label="Classical (Observed Data)", s=60)
        ax.plot(days, regression_melt, color="red", label="Regression Fit", linewidth=2)
        ax.plot(future_days, future_pred, color="green", linestyle="--", label="Future Prediction")

        ax.set_title("Glacier Melt Mapping: Classical vs Regression", fontsize=14)
        ax.set_xlabel("Days")
        ax.set_ylabel("Melt Volume (arbitrary units)")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)


    # Call inside Streamlit
    glacier_mapping_regression_classical()
    import streamlit as st
    import numpy as np
    import matplotlib.pyplot as plt


    def simulated_glacier_map():
        st.markdown("## ❄️ Simulated Glacier Map")

        # Create grid
        x = np.linspace(-5, 5, 200)
        y = np.linspace(-5, 5, 200)
        X, Y = np.meshgrid(x, y)

        # Simulated glacier surface (Gaussian hump + noise)
        Z = np.exp(-(X ** 2 + Y ** 2) / 10) * 100 + np.random.normal(0, 2, X.shape)

        # Plot
        fig, ax = plt.subplots(figsize=(6, 6))
        contour = ax.contourf(X, Y, Z, cmap="Blues")
        fig.colorbar(contour, ax=ax, label="Glacier Thickness (m)")
        ax.set_title("Simulated Glacier Surface Map")

        st.pyplot(fig)


    # Call inside Streamlit
    simulated_glacier_map()
    # pip install cartopy
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature


    def glacier_world_heatmap():
        fig = plt.figure(figsize=(12, 6))
        ax = plt.axes(projection=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor="lightgrey")
        ax.add_feature(cfeature.BORDERS, linestyle=":")
        ax.add_feature(cfeature.COASTLINE)

        # Example glacier coordinates (Himalayas, Alps, Andes, Greenland, Antarctica)
        glacier_coords = [
            (28, 85),  # Himalayas (Nepal/India)
            (46, 9),  # Alps
            (-32, -70),  # Andes
            (72, -40),  # Greenland
            (-75, 0)  # Antarctica
        ]

        for lat, lon in glacier_coords:
            ax.plot(lon, lat, "bo", markersize=8, transform=ccrs.PlateCarree())

        ax.set_title("World Glaciers Map", fontsize=16)
        st.pyplot(fig)


    # Call inside Streamlit
    glacier_world_heatmap()
    import streamlit as st
    import folium
    from streamlit_folium import st_folium


    def dynamic_glacier_map():
        st.markdown("## 🌍 Dynamic Glacier World Map")

        # Center map on Earth
        m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

        # Example glacier locations (Himalayas, Alps, Andes, Greenland, Antarctica, Alaska, etc.)
        glacier_locations = {
            "Himalayas": [28, 85],
            "Alps": [46, 9],
            "Andes": [-32, -70],
            "Greenland": [72, -40],
            "Antarctica": [-75, 0],
            "Alaska": [61, -150]
        }

        # Add markers for glaciers
        for name, coords in glacier_locations.items():
            folium.Marker(
                location=coords,
                popup=f"❄️ {name} Glacier Region",
                icon=folium.Icon(color="blue", icon="cloud")
            ).add_to(m)

        # Render in Streamlit
        st_data = st_folium(m, width=800, height=500)


    # Call inside your Streamlit app
    dynamic_glacier_map()



elif choice == "Glacier Tracker":
    st.markdown(f"<div class='big-title'>Computer Vision <span style='color:{PRIMARY}'>Glacier Tracker</span></div>",
                unsafe_allow_html=True)

    mode = st.radio("Choose data mode", ["Image Pair", "Video"], horizontal=True)
    method = st.selectbox("Segmentation method", ["HSV", "KMeans (auto)"])

    if method == "HSV":
        with st.expander("HSV Thresholds", expanded=True):
            h_min = st.slider("Hue min", 0, 179, 0)
            s_max = st.slider("Saturation max", 0, 255, 60)
            v_min = st.slider("Value min", 0, 255, 150)
        params = SegmentParams(h_min=h_min, s_max=s_max, v_min=v_min)
    else:
        params = SegmentParams()

    if mode == "Image Pair":
        col_a, col_b = st.columns(2)
        with col_a:
            f1 = st.file_uploader("Upload BEFORE image", type=["jpg", "jpeg", "png"])
        with col_b:
            f2 = st.file_uploader("Upload AFTER image", type=["jpg", "jpeg", "png"])

        if f1 and f2:
            img1 = load_image(f1)
            img2 = load_image(f2)
            if method == "HSV":
                m1 = segment_ice_hsv(img1, params)
                m2 = segment_ice_hsv(img2, params)
            else:
                m1 = segment_ice_kmeans(img1)
                m2 = segment_ice_kmeans(img2)
            r1 = area_ratio(m1)
            r2 = area_ratio(m2)
            delta = r2 - r1

            st.markdown("### Visuals")
            ca, cb = st.columns(2)
            with ca:
                st.image(to_rgb(annotate_mask(img1, m1)), caption=f"BEFORE — Ice area: {r1:.2f}%",
                         use_column_width=True)
            with cb:
                st.image(to_rgb(annotate_mask(img2, m2)), caption=f"AFTER — Ice area: {r2:.2f}%", use_column_width=True)

            st.markdown("### Metrics")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("Before ice %", f"{r1:.2f}%")
            mcol2.metric("After ice %", f"{r2:.2f}%")
            mcol3.metric("Change", f"{delta:+.2f}%")

            df_pair = pd.DataFrame({"stage": ["Before", "After"], "ice_%": [r1, r2]})
            fig = px.bar(df_pair, x="stage", y="ice_%", title="Ice Coverage Change")
            st.plotly_chart(fig, use_container_width=True)

            csv = df_pair.to_csv(index=False).encode("utf-8")
            st.download_button("Download metrics CSV", data=csv, file_name="glacier_image_pair_metrics.csv",
                               mime="text/csv")

    else:  # Video mode
        fv = st.file_uploader("Upload glacier video (.mp4/.mov/.avi)", type=["mp4", "mov", "avi"])
        sample_every = st.slider("Sample every N frames", 1, 30, 10)
        max_frames = st.slider("Max sampled frames", 50, 1000, 400, step=50)
        if fv:
            with st.spinner("Processing video frames…"):
                df, previews = process_video(fv, "HSV" if method == "HSV" else "KMeans", params, sample_every,
                                             max_frames)
            if df.empty:
                st.warning("No frames processed. Try reducing 'Sample every N frames' or upload a different video.")
            else:
                st.markdown("### Time-Series of Ice Coverage")
                fig = px.line(df, x="time_s", y="ice_%", markers=True, title="Estimated Ice Area Over Time")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown("### Sampled Frames (Original • Mask • Overlay)")
                grid = st.columns(3)
                for i, (orig, mask, overlay) in enumerate(previews):
                    with grid[i % 3]:
                        st.image(orig, caption=f"Original #{i + 1}", use_column_width=True)
                        st.image(mask, caption=f"Mask #{i + 1}", use_column_width=True)
                        st.image(overlay, caption=f"Overlay #{i + 1}", use_column_width=True)

                # Stats
                st.markdown("### Summary")
                st.dataframe(df)
                st.metric("Avg ice %", f"{df['ice_%'].mean():.2f}%")
                st.metric("Min ice %", f"{df['ice_%'].min():.2f}%")
                st.metric("Max ice %", f"{df['ice_%'].max():.2f}%")

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download time-series CSV", data=csv, file_name="glacier_video_metrics.csv",
                                   mime="text/csv")
    import streamlit as st
    import pandas as pd
    import plotly.express as px

    st.set_page_config(page_title="Glacier Melting Dashboard", layout="wide")

    st.title("Glacier Melting Over Time")
    st.markdown("""
      Upload your CSV file containing **Year** and **Area** columns to visualize glacier melting.
      Use the sliders and hover features for detailed analysis.
      """)

    # Upload CSV
    uploaded_file = st.file_uploader("Upload CSV (columns: Year, Area)", type=["csv"])

    if uploaded_file:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        # Check required columns
        if not {'Year', 'Area'}.issubset(df.columns):
            st.error("CSV must contain 'Year' and 'Area' columns.")
        else:
            # Display data
            st.subheader("Data Preview")
            st.dataframe(df)

            # Convert Year to integer if needed
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce').dropna().astype(int)

            # Sorting data by Year
            df = df.sort_values('Year')

            # Filters
            min_year, max_year = df['Year'].min(), df['Year'].max()
            year_range = st.slider("Select Year Range", min_year, max_year, (min_year, max_year))
            df_filtered = df[(df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1])]

            # Plot
            st.subheader("Glacier Area Over Time")
            fig = px.line(
                df_filtered,
                x='Year',
                y='Area',
                markers=True,
                title='Glacier Area Trend',
                labels={'Area': 'Glacier Area (sq km)', 'Year': 'Year'}
            )
            fig.update_traces(mode='lines+markers')
            fig.update_layout(hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

            # Show filtered data
            st.subheader("Filtered Data")
            st.dataframe(df_filtered)

    else:
        st.info("Please upload your CSV file with 'Year' and 'Area' columns to begin.")
    # Glacier Melting — Portfolio Streamlit App
    # Single-file Streamlit app combining a colorful personal portfolio
    # and a computer-vision pipeline to track glacier melting between
    # two images (before/after).
    # Run: `streamlit run glacier_portfolio_app.py`

    import streamlit as st
    from PIL import Image, ImageOps
    import numpy as np
    import cv2
    import io
    import base64
    import os
    import tempfile
    from datetime import datetime

    st.set_page_config(page_title="Glacier Portfolio & CV", layout="wide",
                       initial_sidebar_state="expanded")

    # ---------- Styles ----------
    st.markdown(
        """
        <style>
        .header {background: linear-gradient(90deg,#0f172a,#0ea5e9); padding:30px; border-radius:12px; color:white}
        .sub {color: #e2e8f0}
        .card {background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02)); padding:18px; border-radius:12px; box-shadow: 0 8px 20px rgba(2,6,23,0.4)}
        .sm {font-size:0.9rem; color:#cbd5e1}
          .sub{color:green;font-size:large}

        </style>
        """, unsafe_allow_html=True)



    # ---------- Sidebar controls ----------
    st.sidebar.title("Tracker Controls")
    alignment_algo = st.sidebar.selectbox("Alignment algorithm", options=["ORB+Homography", "ECC (if available)"],
                                          index=0)
    use_segmentation = st.sidebar.checkbox("Try semantic segmentation (DeepLabV3 if available)", value=False)
    colormap = st.sidebar.selectbox("Diff colormap", options=["JET", "INFERNO", "HOT", "PLASMA"], index=0)
    thresholding = st.sidebar.slider("Diff threshold (for mask)", 1, 255, 30)


    # ---------- Utility functions ----------

    def pil_to_cv(img_pil):
        arr = np.array(img_pil.convert('RGB'))
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


    def cv_to_pil(img_cv):
        rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)


    def read_image(uploaded_file):
        image = Image.open(uploaded_file)
        return image


    def resize_keeping_aspect(img_cv, max_dim=1200):
        h, w = img_cv.shape[:2]
        scale = min(max_dim / w, max_dim / h, 1.0)
        if scale < 1.0:
            return cv2.resize(img_cv, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img_cv


    # ---------- Alignment functions ----------

    def align_orb(img1, img2, max_features=5000, good_match_percent=0.15):
        """Align img2 to img1 using ORB + Homography."""
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        orb = cv2.ORB_create(max_features)
        k1, d1 = orb.detectAndCompute(img1_gray, None)
        k2, d2 = orb.detectAndCompute(img2_gray, None)
        if d1 is None or d2 is None:
            return img2, None
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(d1, d2)
        matches = sorted(matches, key=lambda x: x.distance)
        numGood = int(len(matches) * good_match_percent)
        matches = matches[:max(numGood, 4)]
        if len(matches) < 4:
            return img2, None
        pts1 = np.zeros((len(matches), 2), dtype=np.float32)
        pts2 = np.zeros((len(matches), 2), dtype=np.float32)
        for i, m in enumerate(matches):
            pts1[i, :] = k1[m.queryIdx].pt
            pts2[i, :] = k2[m.trainIdx].pt
        m, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)
        if m is None:
            return img2, None
        h, w = img1.shape[:2]
        aligned = cv2.warpPerspective(img2, m, (w, h))
        return aligned, m


    def compute_diff(img1, img2, thresh=30):
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        d = cv2.absdiff(g1, g2)
        _, mask = cv2.threshold(d, thresh, 255, cv2.THRESH_BINARY)
        return d, mask


    def apply_colormap(diff_gray, cmap_name="JET"):
        cmap_map = {
            'JET': cv2.COLORMAP_JET,
            'INFERNO': cv2.COLORMAP_INFERNO,
            'HOT': cv2.COLORMAP_HOT,
            'PLASMA': cv2.COLORMAP_PLASMA
        }
        cmap = cmap_map.get(cmap_name, cv2.COLORMAP_JET)
        colored = cv2.applyColorMap(diff_gray, cmap)
        return colored


    # Optional semantic segmentation loader
    seg_model = None
    if use_segmentation:
        try:
            import torch
            import torchvision

            seg_model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True).eval()
            st.sidebar.success("Loaded DeepLabV3 for segmentation (CPU).")
        except Exception as e:
            st.sidebar.warning("Could not load segmentation model: {}".format(e))
            seg_model = None


    def segment_image_pil(pil_img):
        """Return a mask (0/255) from DeepLabV3 for 'person'/'background' style segmentation - good for landscapes too."""
        if seg_model is None:
            return None
        img = pil_img.convert('RGB')
        tf = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
        input_tensor = tf(img).unsqueeze(0)
        with torch.no_grad():
            out = seg_model(input_tensor)['out'][0]
        labels = out.argmax(0).byte().cpu().numpy()
        # Convert labels to binary mask: any label > 0 considered foreground
        mask = (labels > 0).astype(np.uint8) * 255
        return mask


    # ---------- Interface: Upload images ----------
    st.header("Glacier Melting Tracker")
    st.write(
        "Upload a *BEFORE* and *AFTER* image (satellite / drone / photo). The app will align, compute differences and show an interactive before/after comparison.")

    col1, col2 = st.columns(2)
    with col1:
        before_file = st.file_uploader("Upload BEFORE image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], key='before')
    with col2:
        after_file = st.file_uploader("Upload AFTER image", type=['png', 'jpg', 'jpeg', 'tif', 'tiff'], key='after')

    use_sample = False
    if not before_file or not after_file:
        st.info("You can try sample glacier images if you don't have your own. (Small images for demo.)")
        if st.button("Load sample images"):
            use_sample = True
            # Load from package - create two synthetic demo images
            demo_dir = tempfile.gettempdir()
            b = Image.new('RGB', (900, 600), (150, 180, 220))
            a = Image.new('RGB', (900, 600), (150, 180, 220))
            # Draw a white 'ice' blob that shrinks in AFTER
            cvb = pil_to_cv(b)
            cva = pil_to_cv(a)
            cv2.circle(cvb, (450, 300), 200, (240, 240, 255), -1)
            cv2.circle(cva, (450, 300), 150, (240, 240, 255), -1)
            before_img = cvb
            after_img = cva
        else:
            before_img = None
            after_img = None
    else:
        before_img = pil_to_cv(read_image(before_file))
        after_img = pil_to_cv(read_image(after_file))

    if before_img is not None and after_img is not None:
        # Resize to reasonable size
        before_img = resize_keeping_aspect(before_img, max_dim=1200)
        after_img = resize_keeping_aspect(after_img, max_dim=1200)

        st.markdown("### Alignment & Difference")
        colA, colB = st.columns([1, 1])
        with colA:
            st.image(cv_to_pil(before_img), caption='BEFORE (aligned to original orientation)')
        with colB:
            st.image(cv_to_pil(after_img), caption='AFTER (raw upload)')

        # Align
        if alignment_algo == "ORB+Homography":
            aligned_after, H = align_orb(before_img, after_img)
        else:
            aligned_after, H = align_orb(before_img, after_img)

        if H is None:
            st.warning(
                "Could not compute a robust homography — showing unaligned AFTER image. Try images with overlapping regions or different algorithm settings.")
            aligned_after = after_img
        else:
            st.success("Alignment succeeded. Homography matrix computed.")

        st.write("**Homography (first 3x3 block):**")
        st.code(np.array_str(H, precision=3)) if H is not None else None

        st.markdown("---")
        diff_gray, mask = compute_diff(before_img, aligned_after, thresh=thresholding)
        heat = apply_colormap(diff_gray, cmap_name=colormap)

        # Optionally refine mask with segmentation
        if use_segmentation and seg_model is not None:
            try:
                seg_mask_before = segment_image_pil(cv_to_pil(before_img))
                seg_mask_after = segment_image_pil(cv_to_pil(aligned_after))
                if seg_mask_before is not None and seg_mask_after is not None:
                    seg_mask = cv2.bitwise_and(seg_mask_before, seg_mask_after)
                    mask = cv2.bitwise_and(mask, seg_mask)
                    st.sidebar.info("Refined diff mask using DeepLabV3 segmentation.")
            except Exception as e:
                st.sidebar.warning("Segmentation refinement failed: {}".format(e))

        # Overlay heat on top of BEFORE image for visualization
        overlay = cv2.addWeighted(before_img, 0.6, heat, 0.4, 0)
        overlay_pil = cv_to_pil(overlay)
        heat_pil = cv_to_pil(heat)

        # Display interactive before-after slider via simple HTML
        st.markdown("**Interactive comparison (drag):**")


        # Prepare images as base64
        def pil_to_b64(img_pil):
            buffered = io.BytesIO()
            img_pil.save(buffered, format="PNG")
            return base64.b64encode(buffered.getvalue()).decode()


        b64_before = pil_to_b64(cv_to_pil(before_img))
        b64_after = pil_to_b64(overlay_pil)

        slider_html = f"""
        <style>
        .comp-wrap {{width:100%; max-width:1100px; margin:auto}}
        .comp-img {{width:100%; display:block}}
        .comp-slider {{-webkit-appearance: none; width:100%;}}
        </style>
        <div class="comp-wrap">
          <div style="position:relative;">
            <img src="data:image/png;base64,{b64_before}" class="comp-img" id="img1">
            <img src="data:image/png;base64,{b64_after}" class="comp-img" id="img2" style="position:absolute; top:0; left:0; clip:rect(0px,600px,9999px,0px);">
          </div>
          <input type="range" min="0" max="100" value="50" id="s" class="comp-slider">
        </div>
        <script>
        const s = document.getElementById('s');
        const img2 = document.getElementById('img2');
        s.oninput = function(){{
          const val = this.value/100.0;
          const w = img2.naturalWidth;
          const clipx = Math.round(w * val);
          img2.style.clip = 'rect(0px,'+clipx+'px,9999px,0px)';
        }}
        </script>
        """

        st.components.v1.html(slider_html, height=520)

        st.markdown("---")
        st.subheader("Diff Mask & Statistics")
        st.image(cv_to_pil(mask), caption='Binary diff mask (areas of change)')
        # Statistics: percentage changed
        pct = (np.count_nonzero(mask) / mask.size) * 100.0
        st.metric("Percent changed (approx)", f"{pct:.4f}%")


        # Download result as ZIP with images & mask
        def make_download_zip():
            import zipfile
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w') as z:
                for name, img in [('before.png', cv_to_pil(before_img)),
                                  ('after_aligned.png', cv_to_pil(aligned_after)),
                                  ('overlay.png', overlay_pil), ('mask.png', cv_to_pil(mask))]:
                    b = io.BytesIO()
                    img.save(b, format='PNG')
                    z.writestr(name, b.getvalue())
            return buf.getvalue()


        zipped = make_download_zip()
        st.download_button("Download results (zip)", data=zipped,
                           file_name=f"glacier_results_{datetime.now().strftime('%Y%m%d_%H%M')}.zip")

    else:
        st.info("Upload both BEFORE and AFTER images (or load sample) to run the tracker.")
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np

    frames = np.arange(1, 21)
    np.random.seed(42)
    initial_area = 1000
    melting = np.cumsum(np.random.uniform(5, 20, size=20))
    areas = np.clip(initial_area - melting, 0, None)

    fig, ax = plt.subplots()
    ax.plot(frames, areas, marker='o', color='blue')
    ax.set_title("Simulated Glacier Melting Tracking")
    ax.set_xlabel("Frame / Time Step")
    ax.set_ylabel("Glacier Area (arbitrary units)")
    ax.grid(True)

    st.pyplot(fig)
    import streamlit as st
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    years = np.arange(2000, 2021)
    np.random.seed(123)
    initial_value = 1000
    annual_change = np.random.uniform(-50, -10, size=len(years))
    values = initial_value + np.cumsum(annual_change)
    values = np.clip(values, 0, None)

    df = pd.DataFrame({"Year": years, "Value": values})

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["Year"], df["Value"], marker='o', linestyle='-', color='teal')
    ax.set_title("Yearly Tracking Data Flow (Simulated)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Tracked Value (e.g., Glacier Area)")
    ax.grid(True)
    plt.xticks(years, rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    import streamlit as st
    import pandas as pd
    from PIL import Image

    st.title("📷 Glacier Image Viewer with Tracking")

    # --- Upload images dynamically ---
    uploaded_files = st.file_uploader(
        "Upload Glacier Images",
        type=["tif", "tiff", "png", "jpg", "jpeg"],
        accept_multiple_files=True
    )

    # --- Initialize session state for view counts ---
    if "views" not in st.session_state:
        st.session_state.views = {}

    # --- If files uploaded, show them ---
    if uploaded_files:
        for file in uploaded_files:
            # If new file, add it to session state views
            if file.name not in st.session_state.views:
                st.session_state.views[file.name] = 0

            # Load image
            img = Image.open(file)
            st.image(img, caption=file.name, use_column_width=True)

            # Button to track "views"
            if st.button(f"View {file.name}"):
                st.session_state.views[file.name] += 1
                st.success(f"Viewed {file.name}!")

    # --- Show current view counts ---
    if st.session_state.views:
        st.subheader("📊 View Counts")
        df_views = pd.DataFrame.from_dict(st.session_state.views, orient='index', columns=['Views'])
        st.bar_chart(df_views)

        # Optionally show raw table
        st.subheader("📋 Raw Data")
        st.dataframe(df_views)
    else:
        st.info("📂 Please upload glacier images to start tracking views.")

    import streamlit as st
    import pandas as pd
    import plotly.express as px

    # Sample glacier location data (latitude, longitude, name)
    data = {
        "Name": ["Glacier A", "Glacier B", "Glacier C"],
        "Latitude": [61.5, 46.8, 78.9],
        "Longitude": [-149.9, 11.2, 16.0],
        "Size_km2": [120, 80, 200]
    }
    df = pd.DataFrame(data)

    st.title("World Glacier Map")

    fig = px.scatter_geo(df,
                         lat="Latitude",
                         lon="Longitude",
                         hover_name="Name",
                         size="Size_km2",
                         projection="natural earth",
                         title="Sample Glacier Locations")
    st.plotly_chart(fig)
    # --------------------------------------------------------------

    # ---------- Footer / Extras ----------
    st.markdown("---")
    st.markdown("### Notes & Next steps")
    st.write(
        "- This demo performs classical CV alignment and simple differencing. For production-grade glacier change detection, consider using multispectral satellite data (Landsat/Sentinel), radiometric corrections, image co-registration with georeferencing, and specialized change detection algorithms.\n- Want an .ipynb notebook version, or to add a GIS map overlay (georeferenced), or to run GPU-accelerated segmentation? Tell me and I will adapt it.")

    # Helpful quick links for running locally
    st.markdown(
        "**Run locally:** `pip install streamlit opencv-python pillow numpy` then `streamlit run glacier_portfolio_app.py`")
    import streamlit as st



elif choice == "Portfolio":
    st.markdown(f"<div class='big-title'>My <span style='color:{PRIMARY}'>Portfolio</span></div>",
                unsafe_allow_html=True)
    # ---------- Header / Portfolio ----------
    with st.container():
        left, right = st.columns([3, 1])
        with left:
            st.header("🧊 Glacier Melting Tracker Using Computer Vision")
            st.markdown('<div class="header">', unsafe_allow_html=True)
            st.title("Vishesh Kumar Prajapati — Computer Vision & Data Science")
            st.markdown("<p  class='sub' >Portfolio · Computer Vision · Python · Streamlit</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("---")
            st.header("About Me")
            st.write(
                "I build interactive visualisations, CV pipelines and deploy them as web apps. This single-file Streamlit app demonstrates a colorful portfolio section and a glacier melting tracker using OpenCV.")
            st.markdown("**Skills:** Python, OpenCV, PyTorch (optional), Streamlit, Remote Sensing, Data Viz")
            st.markdown("---")
            st.header("Projects & Contacts")
            st.markdown(
                "- **Glacier Melting Tracker** — Image alignment, differencing, optional segmentation, interactive before/after slider.")
            st.markdown("- **Other projects** — Weather bot, Deforestation sensor, Movie UI, Salary prediction model.")
        st.subheader("📬 Contact Details")

        st.markdown("""
            [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/vishesh-kumar-prajapati-45111829a?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app)  
            [![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/vishes-i/future-interns/commit/ddc7ef65bcf8417b718241c2fe8d6dd715d8a8b6)  
            [![Email](https://img.shields.io/badge/Email-Send-red?logo=gmail)](mailto:Visheshprajapati7920@gmail.com)
            """)
        with right:
            st.image(
                "https://t3.ftcdn.net/jpg/10/25/12/36/360_F_1025123627_aiwBz4jP8ED17Tr2ut8XxugPo69MW61J.jpg",
                # Placeholder image URL
                caption="SHIVSHAKTI",
                width=200
            )

    st.markdown("---")
    # --- SKILLS / BADGES SECTION ---
    st.markdown("## 🛠️ Skills & Technologies")

    st.markdown(
        """
        <div style='display: flex; flex-wrap: wrap; gap: 10px;'>
            <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Numpy-013243?logo=numpy&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/Matplotlib-00457C?logo=matplotlib&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black&style=for-the-badge" height="30">
            <img src="https://img.shields.io/badge/MySQL-4479A1?logo=mysql&logoColor=white&style=for-the-badge" height="30">
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")
    import streamlit as st


    def badges_section():
        st.markdown("## 🏅 Achievements & Badges")

        badges = [
            {
                "name": "Machine Learning Specialization",
                "issuer": "Coursera / Andrew Ng",
                "year": "2023",
                "link": "https://www.coursera.org/specializations/machine-learning-introduction"
            },
            {
                "name": "Deep Learning Professional Certificate",
                "issuer": "DeepLearning.AI",
                "year": "2024",
                "link": "https://www.deeplearning.ai/"
            },
            {
                "name": "Data Science with Python",
                "issuer": "Kaggle",
                "year": "2024",
                "link": "https://www.kaggle.com/learn"
            },
            {
                "name": "Streamlit Creator Badge",
                "issuer": "Streamlit Community",
                "year": "2025",
                "link": "https://discuss.streamlit.io/"
            }
        ]

        # Display badges in a grid
        cols = st.columns(2)
        for i, badge in enumerate(badges):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="padding:15px; border-radius:12px; background:#f5f5f5; margin:10px 0;">
                    <h4>🏅 {badge['name']}</h4>
                    <p><b>Issuer:</b> {badge['issuer']}</p>
                    <p><b>Year:</b> {badge['year']}</p>
                    <a href="{badge['link']}" target="_blank">🔗 View Credential</a>
                </div>
                """, unsafe_allow_html=True)


    # Call inside Streamlit app
    badges_section()

    st.markdown(
        "<p class='subtitle'>Selected projects showcasing data science, computer vision, and ML engineering.</p>",
        unsafe_allow_html=True)


    def project_section():
        st.markdown("## 🚀 Project Portfolio")

        projects = [
            {
                "title": "🧊 Glacier Monitoring",
                "desc": "A computer vision + SfM inspired system to track glacier melt using image differencing.",
                "features": [
                    "Real-time glacier melt tracking",
                    "SfM-style 3D reconstruction simulation",
                    "Interactive melt trend graphs"
                ]
            },
            {
                "title": "💧 Water Quality Prediction",
                "desc": "Machine learning model to predict water potability based on chemical parameters.",
                "features": [
                    "Random Forest regression/classification",
                    "Dynamic data upload & preprocessing",
                    "Interactive prediction dashboard"
                ]
            },
            {
                "title": "💰 Salary Prediction System",
                "desc": "ML model that predicts salaries based on demographics & work attributes.",
                "features": [
                    "Random Forest regression",
                    "User-friendly inputs",
                    "Real-time salary graphs"
                ]
            }
        ]

        # Interactive project display
        for proj in projects:
            with st.expander(proj["title"], expanded=False):
                st.write(proj["desc"])
                st.markdown("**Key Features:**")
                for f in proj["features"]:
                    st.markdown(f"- {f}")
                st.markdown("---")


    # Call inside Streamlit app
    project_section()
    import streamlit as st


    def experience_and_skills():
        st.markdown("## 💼 Experience & Skills")

        # Tabs for separation
        tab1, tab2 = st.tabs(["💼 Experience", "🛠️ Skills"])

        # --- EXPERIENCE TAB ---
        with tab1:
            st.subheader("💼 Professional Experience")

            exp_list = [
                {
                    "role": "Machine Learning Engineer",
                    "company": "AI Research Lab",
                    "duration": "2025 - Present",
                    "details": "Working on computer vision projects for glacier monitoring and climate change."
                },
                {
                    "role": "Data Scientist",
                    "company": "Tech Solutions Inc.",
                    "duration": "2023 - 2024",
                    "details": "Built ML pipelines for salary prediction, water quality prediction, and dashboard analytics."
                },
                {
                    "role": "Intern",
                    "company": "Open Source Projects",
                    "duration": "2024 - 2025",
                    "details": "Contributed to open-source AI tools and collaborated with researchers worldwide."
                }
            ]

            for exp in exp_list:
                with st.expander(f"📌 {exp['role']} @ {exp['company']} ({exp['duration']})"):
                    st.write(exp["details"])

        # --- SKILLS TAB ---
        with tab2:
            st.subheader("🛠️ Technical Skills")

            skills = {
                "Python & Machine Learning": 90,
                "Computer Vision (OpenCV, Torch)": 85,
                "Data Science & Visualization": 80,
                "Streamlit / Dashboards": 75,
                "Deep Learning (PyTorch, TensorFlow)": 70,
                "SQL & Data Engineering": 65,
            }

            # Display in two columns
            col1, col2 = st.columns(2)

            for i, (skill, level) in enumerate(skills.items()):
                if i % 2 == 0:
                    col1.write(f"**{skill}**")
                    col1.progress(level)
                else:
                    col2.write(f"**{skill}**")
                    col2.progress(level)


    # Call inside Streamlit app
    experience_and_skills()




import streamlit as st
import json
from PIL import Image, ImageDraw, ImageOps
import requests
from io import BytesIO

# ---------- Function to Make Circular Images ----------
def image_circle(img_source):
    """
    Takes either a URL or an uploaded file and returns a circular PIL image.
    """
    try:
        # If it's a URL string
        if isinstance(img_source, str):
            response = requests.get(img_source)
            img = Image.open(BytesIO(response.content)).convert("RGBA")
        else:
            # If it's a file uploaded by user
            img = Image.open(img_source).convert("RGBA")

        size = min(img.size)
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)

        output = ImageOps.fit(img, (size, size))
        output.putalpha(mask)
        return output

    except Exception as e:
        st.warning(f"⚠️ Could not process image: {e}")
        return None


# ---------- TEAM SECTION ----------
if choice == "Team":  # ✅ Correct start of Team section
    st.markdown(f"<div class='big-title'>Meet the <span style='color:blue'>Team</span></div>", unsafe_allow_html=True)
    st.markdown("""
    <h2 style='text-align: center; 
               font-size: 36px; 
               background: -webkit-linear-gradient(45deg, #FF6B6B, #5F27CD); 
               -webkit-background-clip: text; 
               -webkit-text-fill-color: transparent;'>
    ✨ Meet Our Amazing Team ✨
    </h2>
    """, unsafe_allow_html=True)

    if "team" not in st.session_state:
        st.session_state.team = [
            {"name": "Vishesh Kumar Prajapati", "role": "Founder / ML Engineer",
             "bio": "Passionate ML Engineer building intelligent systems and dashboards.",
             "email": "visheshprajapati7920@gmail.com",
             "linkedin": "https://www.linkedin.com/in/vishesh-kumar-prajapati-45111829a",
             "github": "https://github.com/vishes-i",
             "photo_url": "https://d2gg9evh47fn9z.cloudfront.net/1600px_COLOURBOX37232552.jpg"},

            {"name": "Sumit Yadav", "role": "Web Developer",
             "bio": "Frontend & backend developer who loves crafting responsive web apps.",
             "email": "sy2902913@gmail.com",
             "linkedin": "https://www.linkedin.com/in/sumit-yadav-3b93a92a9",
             "github": "https://github.com/",
             "photo_url": "https://d2gg9evh47fn9z.cloudfront.net/1600px_COLOURBOX37236066.jpg"},

            {"name": "Tejashwani Singh Rathore", "role": "Web Developer",
             "bio": "Frontend developer who loves crafting responsive web apps.",
             "email": "tejaswanirathore910@gmail.com",
             "linkedin": "https://www.linkedin.com/in/tejaswanirathore-3b93a92a9",
             "github": "https://github.com/",
             "photo_url": "https://img.freepik.com/premium-photo/young-girl-hr-3d-character-young-working-girl-cartoon-character-professional-girl-character_1002350-2145.jpg?w=2000"},

            {"name": "Vijay Kharwar", "role": "Web Developer",
             "bio": "Frontend developer who loves crafting responsive web apps.",
             "email": "vijaykharwargzp2003@gmail.com",
             "linkedin": "https://www.linkedin.com/in/vijay-kharwar-b290aa2ab",
             "github": "https://github.com/vijaykharwargzp2003-coder",
             "photo_url": "https://img.freepik.com/free-vector/confident-businessman-with-smile_1308-134106.jpg"}
        ]

    # ---- Display Team ----
    for idx, member in enumerate(st.session_state.team):
        with st.container():
            col1, col2 = st.columns([1, 4])

            with col1:
                img = image_circle(member.get("photo_url") or member.get("photo"))
                if img:
                    st.image(img, use_container_width=False)
                else:
                    st.image(member.get("photo_url") or member.get("photo"), use_container_width=False)

            with col2:
                st.subheader(member["name"])
                st.caption(member["role"])
                st.write(member.get("bio", "No biography added yet."))

                c1, c2, c3 = st.columns(3)
                with c1:
                    if member.get("email"):
                        st.link_button("📧 Email", f"mailto:{member['email']}")
                with c2:
                    if member.get("linkedin"):
                        st.link_button("💼 LinkedIn", member["linkedin"])
                with c3:
                    if member.get("github"):
                        st.link_button("💻 GitHub", member["github"])

                # Edit Section
                with st.expander("✏️ Edit Member"):
                    with st.form(f"edit_form_{idx}"):
                        member["name"] = st.text_input("Name", member["name"])
                        member["role"] = st.text_input("Role", member["role"])
                        member["bio"] = st.text_area("Biography", member.get("bio", ""))
                        member["email"] = st.text_input("Email", member["email"])
                        member["linkedin"] = st.text_input("LinkedIn", member["linkedin"])
                        member["github"] = st.text_input("GitHub", member["github"])
                        member["photo_url"] = st.text_input("Photo URL", member.get("photo_url", ""))
                        new_photo = st.file_uploader("Upload Photo", type=["png", "jpg", "jpeg"], key=f"photo_{idx}")
                        update = st.form_submit_button("✅ Update")
                        if update:
                            if new_photo:
                                member["photo"] = new_photo
                            st.success(f"Updated {member['name']}")

    # ---- Add New Member ----
    with st.expander("➕ Add New Member"):
        with st.form("add_member_form"):
            st.subheader("Add a New Team Member")
            name = st.text_input("Name")
            role = st.text_input("Role")
            bio = st.text_area("Biography")
            email = st.text_input("Email")
            linkedin = st.text_input("LinkedIn")
            github = st.text_input("GitHub")
            photo_url = st.text_input("Photo URL")
            photo = st.file_uploader("Upload Photo", type=["png", "jpg", "jpeg"], key="new_photo")
            add = st.form_submit_button("Add Member")
            if add and name and role:
                new_member = {"name": name, "role": role, "bio": bio, "email": email,
                              "linkedin": linkedin, "github": github, "photo_url": photo_url, "photo": photo}
                st.session_state.team.append(new_member)
                st.success(f"Added {name} to the team!")

    # ---- Delete Member ----
    with st.form("delete_member_form"):
        st.subheader("🗑️ Delete a Member")
        names = [m['name'] for m in st.session_state.team]
        delete_name = st.selectbox("Select member to delete", [""] + names)
        delete = st.form_submit_button("Delete")
        if delete and delete_name:
            st.session_state.team = [m for m in st.session_state.team if m['name'] != delete_name]
            st.success(f"Deleted {delete_name}")

    # ---- Export ----
    st.download_button(
        "📥 Export Team JSON",
        data=json.dumps(st.session_state.team, indent=2, default=str).encode("utf-8"),
        file_name="team.json"
    )

  

elif choice == "About":
    st.markdown(f"<div class='big-title'>About <span style='color:{PRIMARY}'>This App</span></div>",
                unsafe_allow_html=True)
    import streamlit as st


    def about_us_section():
        st.markdown("## ℹ️ About Us")

        # Tabs for better navigation
        tab1, tab2, tab3 = st.tabs(["🌍 Mission", "🚀 Vision", "🤝 Values"])

        with tab1:
            st.subheader("🌍 Our Mission")
            st.write(
                "We are dedicated to leveraging **AI, Computer Vision, and Data Science** "
                "to solve real-world environmental and socio-economic challenges. "
                "From **glacier monitoring** to **salary & water quality prediction**, "
                "our mission is to make impactful solutions accessible to everyone."
            )

        with tab2:
            st.subheader("🚀 Our Vision")
            st.write(
                "To create **open-source, innovative, and scalable** solutions that help "
                "researchers, industries, and policymakers tackle challenges like **climate change**, "
                "**resource management**, and **sustainability**."
            )

        with tab3:
            st.subheader("🤝 Our Values")
            st.markdown("""
            - 🌱 **Sustainability First**: Every project is built with environmental awareness.  
            - 💡 **Innovation**: We integrate cutting-edge techniques in ML & CV.  
            - 📖 **Open Knowledge**: Sharing code, data, and insights for the community.  
            - 🤝 **Collaboration**: Working with researchers and institutions worldwide.  
            """)

        # Bonus: Interactive feedback
        with st.expander("💬 Share Your Thoughts"):
            feedback = st.text_area("What do you think about us?")
            if st.button("Submit Feedback"):
                st.success("✅ Thank you for your feedback!")


    # Call inside Streamlit app
    about_us_section()

    st.markdown(
        """
        This demo estimates glacier/ice coverage using simple image segmentation. Two methods are available:

        - **HSV thresholding**: picks bright, low-saturation pixels (typical for snow/ice). Good for well-lit images.
        - **KMeans (auto)**: clusters colors and assumes the brightest cluster is ice. Robust when lighting varies.

        ⚠️ **Limitations**: This is not a scientific tool. Real-world glacier analysis uses orthorectified imagery, radiometric calibration, cloud masking, terrain shadows handling, and manual QA.

        **Tips**
        - Prefer bird's-eye images with consistent lighting across time.
        - Crop images to the glacier region before uploading for better signal.
        - Keep camera viewpoint consistent when comparing before/after.
        """
    )
