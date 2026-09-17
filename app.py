import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="MobileNet-SSD Object Detection",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM COLOR THEME
# =========================================================

st.markdown(
    """
<style>

    /* =====================================================
       COLOR PALETTE
       Deep Pine  : #0F2E23
       Olive Moss : #798F53
       Warm Ivory : #F4F1DE
       Dusty Rose : #C88582
       Deep Teal  : #1B5B65
       ===================================================== */


    /* Main application */
    .stApp {
        background-color: #F4F1DE;
        color: #0F2E23;
    }

    .main {
        background-color: #F4F1DE;
    }


    /* =====================================================
       SIDEBAR
       ===================================================== */

    section[data-testid="stSidebar"] {
        background-color: #0F2E23;
    }

    section[data-testid="stSidebar"] * {
        color: #F4F1DE !important;
    }


    /* =====================================================
       HEADINGS
       ===================================================== */

    h1 {
        color: #0F2E23 !important;
        font-weight: 800 !important;
    }

    h2 {
        color: #1B5B65 !important;
        font-weight: 750 !important;
    }

    h3 {
        color: #0F2E23 !important;
        font-weight: 700 !important;
    }

    p {
        color: #0F2E23;
    }


    /* =====================================================
       PROJECT HEADER
       ===================================================== */

    .project-header {
        background: linear-gradient(
            135deg,
            rgba(121, 143, 83, 0.14),
            rgba(27, 91, 101, 0.10)
        );

        border: 1px solid rgba(121, 143, 83, 0.45);

        border-radius: 20px;

        padding: 24px 28px;

        margin-bottom: 25px;

        box-shadow:
            0 5px 15px rgba(15, 46, 35, 0.05);
    }


    .project-title {
        color: #0F2E23;

        font-size: 32px;

        font-weight: 800;

        margin-bottom: 7px;
    }


    .project-subtitle {
        color: #1B5B65;

        font-size: 16px;

        margin: 0;
    }


    /* =====================================================
       METRIC CARDS
       ===================================================== */

    div[data-testid="stMetric"] {

        background-color: rgba(121, 143, 83, 0.13);

        border: 1px solid rgba(121, 143, 83, 0.45);

        border-radius: 18px;

        padding: 20px;

        min-height: 120px;

        box-shadow:
            0 4px 12px rgba(15, 46, 35, 0.05);
    }


    div[data-testid="stMetricLabel"] {
        color: #0F2E23 !important;

        font-weight: 600 !important;
    }


    div[data-testid="stMetricValue"] {
        color: #1B5B65 !important;

        font-weight: 800 !important;
    }


    /* =====================================================
       SECTION CARD
       ===================================================== */

    .section-card {

        background-color: rgba(255, 255, 255, 0.50);

        border: 1px solid rgba(121, 143, 83, 0.45);

        border-radius: 18px;

        padding: 20px 24px;

        margin-top: 22px;

        box-shadow:
            0 4px 14px rgba(15, 46, 35, 0.04);
    }


    .section-title {

        color: #1B5B65;

        font-size: 22px;

        font-weight: 750;

        margin-bottom: 16px;
    }


    /* =====================================================
       DETECTED CLASS CARDS
       ===================================================== */

    .detected-class {

        background-color: rgba(27, 91, 101, 0.10);

        border-left: 5px solid #1B5B65;

        border-radius: 12px;

        padding: 13px 18px;

        margin: 9px 0;

        color: #0F2E23;

        font-size: 16px;

        font-weight: 650;
    }


    /* =====================================================
       CONFIDENCE CARDS
       ===================================================== */

    .confidence-card {

        background-color: rgba(121, 143, 83, 0.12);

        border: 1px solid rgba(121, 143, 83, 0.35);

        border-radius: 14px;

        padding: 18px;

        text-align: center;

        min-height: 105px;
    }


    .confidence-card.rose {

        background-color: rgba(200, 133, 130, 0.16);

        border-color: rgba(200, 133, 130, 0.50);
    }


    .confidence-label {

        color: #0F2E23;

        font-size: 14px;

        font-weight: 600;
    }


    .confidence-value {

        color: #1B5B65;

        font-size: 26px;

        font-weight: 800;

        margin-top: 5px;
    }


    .rose-value {
        color: #C88582;
    }


    /* =====================================================
       DOWNLOAD BUTTON
       ===================================================== */

    .stDownloadButton > button {

        background-color: #1B5B65 !important;

        color: #F4F1DE !important;

        border: none !important;

        border-radius: 12px !important;

        padding: 11px 24px !important;

        font-weight: 700 !important;

        transition: 0.2s ease;
    }


    .stDownloadButton > button:hover {

        background-color: #0F2E23 !important;

        color: #F4F1DE !important;
    }


    /* =====================================================
       INFO BOX
       ===================================================== */

    div[data-testid="stAlert"] {

        border-radius: 13px;
    }


    /* =====================================================
       FILE UPLOADER
       ===================================================== */

    section[data-testid="stFileUploaderDropzone"] {

        background-color: rgba(121, 143, 83, 0.08);

        border: 2px dashed #798F53;

        border-radius: 15px;
    }


    /* =====================================================
       IMAGES
       ===================================================== */

    img {
        border-radius: 14px;
    }


    /* =====================================================
       DIVIDER
       ===================================================== */

    hr {
        border-color: rgba(121, 143, 83, 0.35);
    }


    /* =====================================================
       FOOTER
       ===================================================== */

    .footer {

        text-align: center;

        color: #798F53;

        font-size: 13px;

        padding: 30px 0 10px 0;
    }

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# MOBILE-NET SSD CLASSES
# =========================================================

CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]


# =========================================================
# CLASS COLORS FOR BOUNDING BOXES
# =========================================================

CLASS_COLORS = {

    "person": (255, 0, 0),

    "car": (0, 255, 0),

    "dog": (0, 0, 255),

    "bicycle": (255, 255, 0),

    "bottle": (255, 0, 255),

    "chair": (0, 255, 255),

    "bus": (128, 0, 128),

    "motorbike": (0, 165, 255),

    "cat": (128, 128, 0),

    "bird": (255, 128, 0),

    "boat": (255, 165, 0),

    "horse": (0, 128, 255),

    "cow": (128, 255, 0),

    "train": (255, 0, 128),

    "sofa": (128, 128, 255),

    "tvmonitor": (0, 255, 128),

    "pottedplant": (128, 255, 255),

    "diningtable": (255, 128, 128),

    "sheep": (255, 255, 128),

    "aeroplane": (128, 0, 255)
}


DEFAULT_COLOR = (255, 255, 255)


# =========================================================
# LOAD MOBILE-NET SSD MODEL
# =========================================================

@st.cache_resource
def load_model():

    prototxt_path = "deploy.prototxt"

    model_path = "mobilenet_iter_73000.caffemodel"

    net = cv2.dnn.readNetFromCaffe(
        prototxt_path,
        model_path
    )

    return net


net = load_model()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown(
    """
<div style="
    font-size:26px;
    font-weight:800;
    margin-bottom:15px;
">
    🤖 Detection Settings
</div>
""",
    unsafe_allow_html=True
)


st.sidebar.markdown(
    """
<div style="
    font-size:14px;
    line-height:1.6;
    margin-bottom:15px;
">
    Adjust the confidence threshold to control
    which detections are displayed.
</div>
""",
    unsafe_allow_html=True
)


confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.50,
    max_value=1.00,
    value=0.50,
    step=0.05
)


st.sidebar.markdown(
    f"""
<div style="
    background-color:#798F53;
    padding:10px;
    border-radius:10px;
    text-align:center;
    margin-top:10px;
    color:#F4F1DE;
">
    <b>Selected threshold:
    {confidence_threshold * 100:.0f}%</b>
</div>
""",
    unsafe_allow_html=True
)


st.sidebar.info(
    "Objects with confidence below the selected "
    "threshold will not be displayed."
)


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    """
<div class="project-header">

<div class="project-title">
🤖 MobileNet-SSD Object Detection
</div>

<div class="project-subtitle">
Upload an image and detect objects using a pretrained
MobileNet-SSD model.
</div>

</div>
""",
    unsafe_allow_html=True
)


# =========================================================
# IMAGE UPLOADER
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# DETECTION
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # Read uploaded image
    # -----------------------------------------------------

    pil_image = Image.open(
        uploaded_file
    ).convert("RGB")


    image_rgb = np.array(
        pil_image
    )


    image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )


    h, w = image.shape[:2]


    # -----------------------------------------------------
    # Create blob
    # -----------------------------------------------------

    blob = cv2.dnn.blobFromImage(
        cv2.resize(
            image,
            (300, 300)
        ),
        0.007843,
        (300, 300),
        127.5
    )


    # -----------------------------------------------------
    # Run MobileNet-SSD
    # -----------------------------------------------------

    net.setInput(blob)

    detections = net.forward()


    # -----------------------------------------------------
    # Output image
    # -----------------------------------------------------

    output_image = image.copy()


    detected_objects = []

    confidence_scores = []


    # =====================================================
    # PROCESS DETECTIONS
    # =====================================================

    for i in range(
        detections.shape[2]
    ):

        confidence = detections[
            0, 0, i, 2
        ]


        if confidence >= confidence_threshold:

            class_id = int(
                detections[
                    0, 0, i, 1
                ]
            )


            label = CLASSES[
                class_id
            ]


            detected_objects.append(
                label
            )


            confidence_scores.append(
                confidence
            )


            # -------------------------------------------------
            # Bounding box
            # -------------------------------------------------

            box = (
                detections[
                    0, 0, i, 3:7
                ]
                * np.array(
                    [w, h, w, h]
                )
            )


            startX, startY, endX, endY = (
                box.astype("int")
            )


            # Keep box inside image

            startX = max(
                0,
                startX
            )

            startY = max(
                0,
                startY
            )

            endX = min(
                w - 1,
                endX
            )

            endY = min(
                h - 1,
                endY
            )


            # -------------------------------------------------
            # Get class color
            # -------------------------------------------------

            color = CLASS_COLORS.get(
                label,
                DEFAULT_COLOR
            )


            # -------------------------------------------------
            # Draw bounding box
            # -------------------------------------------------

            cv2.rectangle(
                output_image,

                (startX, startY),

                (endX, endY),

                color,

                3
            )


            # -------------------------------------------------
            # Label
            # -------------------------------------------------

            text = (
                f"{label}: "
                f"{confidence * 100:.1f}%"
            )


            (
                text_width,
                text_height
            ), baseline = cv2.getTextSize(
                text,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )


            text_y = max(
                startY - 10,
                text_height + 10
            )


            # Label background

            cv2.rectangle(
                output_image,

                (
                    startX,
                    text_y
                    - text_height
                    - baseline
                ),

                (
                    startX
                    + text_width,

                    text_y
                    + baseline
                ),

                color,

                -1
            )


            # Label text

            cv2.putText(
                output_image,

                text,

                (
                    startX,
                    text_y
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.6,

                (255, 255, 255),

                2
            )


    # =====================================================
    # CONVERT OUTPUT TO RGB
    # =====================================================

    output_rgb = cv2.cvtColor(
        output_image,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # DETECTION RESULT
    # =====================================================

    st.markdown(
        """
<div class="section-title">
🖼️ Detection Result
</div>
""",
        unsafe_allow_html=True
    )


    col1, col2 = st.columns(
        2,
        gap="large"
    )


    with col1:

        st.markdown(
            "### Original Image"
        )

        st.image(
            pil_image,
            use_container_width=True
        )


    with col2:

        st.markdown(
            "### Detected Objects"
        )

        st.image(
            output_rgb,
            use_container_width=True
        )


    st.divider()


    # =====================================================
    # DETECTION SUMMARY
    # =====================================================

    st.markdown(
        """
<div class="section-title">
📊 Detection Summary
</div>
""",
        unsafe_allow_html=True
    )


    if detected_objects:

        object_counts = Counter(
            detected_objects
        )


        # -------------------------------------------------
        # TOP METRICS
        # -------------------------------------------------

        col1, col2, col3, col4 = st.columns(
            4
        )


        with col1:

            st.metric(
                "Total Objects",
                len(detected_objects)
            )


        with col2:

            st.metric(
                "Object Classes",
                len(object_counts)
            )


        with col3:

            st.metric(
                "Average Confidence",
                f"{np.mean(confidence_scores) * 100:.2f}%"
            )


        with col4:

            st.metric(
                "Highest Confidence",
                f"{max(confidence_scores) * 100:.2f}%"
            )


        # =================================================
        # DETECTED CLASSES
        # =================================================

        st.markdown(
            """
<div class="section-card">

<div class="section-title">
🔎 Detected Classes
</div>
""",
            unsafe_allow_html=True
        )


        for label, count in object_counts.items():

            st.markdown(
                f"""
<div class="detected-class">
🔎 &nbsp; {label.capitalize()} — {count}
</div>
""",
                unsafe_allow_html=True
            )


        st.markdown(
            """
</div>
""",
            unsafe_allow_html=True
        )


        # =================================================
        # CONFIDENCE DETAILS
        # =================================================

        st.markdown(
            """
<div class="section-card">

<div class="section-title">
📈 Confidence Details
</div>
""",
            unsafe_allow_html=True
        )


        lowest_confidence = (
            min(confidence_scores) * 100
        )


        average_confidence = (
            np.mean(confidence_scores) * 100
        )


        highest_confidence = (
            max(confidence_scores) * 100
        )


        c1, c2, c3 = st.columns(
            3
        )


        # -------------------------------------------------
        # Lowest confidence
        # -------------------------------------------------

        with c1:

            st.markdown(
                f"""
<div class="confidence-card rose">

<div class="confidence-label">
Lowest Confidence
</div>

<div class="confidence-value rose-value">
{lowest_confidence:.2f}%
</div>

</div>
""",
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # Average confidence
        # -------------------------------------------------

        with c2:

            st.markdown(
                f"""
<div class="confidence-card">

<div class="confidence-label">
Average Confidence
</div>

<div class="confidence-value">
{average_confidence:.2f}%
</div>

</div>
""",
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # Highest confidence
        # -------------------------------------------------

        with c3:

            st.markdown(
                f"""
<div class="confidence-card">

<div class="confidence-label">
Highest Confidence
</div>

<div class="confidence-value">
{highest_confidence:.2f}%
</div>

</div>
""",
                unsafe_allow_html=True
            )


        st.markdown(
            """
</div>
""",
            unsafe_allow_html=True
        )


        # =================================================
        # DOWNLOAD RESULT
        # =================================================

        st.markdown("")


        result_bytes = cv2.imencode(
            ".jpg",
            output_image
        )[1].tobytes()


        st.download_button(
            label="⬇️ Download Detection Result",

            data=result_bytes,

            file_name="mobilenet_ssd_detection_result.jpg",

            mime="image/jpeg"
        )


    else:

        st.warning(
            "No objects were detected above the selected "
            "confidence threshold."
        )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
<div class="footer">
MobileNet-SSD • Computer Vision • Object Detection
</div>
""",
    unsafe_allow_html=True
)
