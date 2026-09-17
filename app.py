import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MobileNet-SSD Object Detection",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CLASS LABELS
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
# DIFFERENT COLORS FOR DIFFERENT CLASSES
# OpenCV uses BGR
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
# LOAD MOBILE NET-SSD MODEL
# =========================================================

@st.cache_resource
def load_model():

    prototxt_path = "deploy.prototxt"
    model_path = "mobilenet_iter_73000.caffemodel"

    # Check model files
    import os

    if not os.path.exists(prototxt_path):
        st.error("deploy.prototxt file not found.")
        st.stop()

    if not os.path.exists(model_path):
        st.error("mobilenet_iter_73000.caffemodel file not found.")
        st.stop()

    try:

        # Primary method
        net = cv2.dnn.readNetFromCaffe(
            prototxt_path,
            model_path
        )

    except AttributeError:

        # Fallback method
        net = cv2.dnn.readNet(
            model_path,
            prototxt_path,
            "Caffe"
        )

    return net


net = load_model()


# =========================================================
# TITLE
# =========================================================

st.title("🤖 MobileNet-SSD Object Detection")

st.write(
    "Upload an image and detect objects using a "
    "pretrained MobileNet-SSD model."
)

st.divider()


# =========================================================
# SIDEBAR SETTINGS
# =========================================================

st.sidebar.title("⚙️ Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.50,
    max_value=1.00,
    value=0.50,
    step=0.05
)

st.sidebar.write(
    f"Selected threshold: "
    f"**{confidence_threshold * 100:.0f}%**"
)

st.sidebar.info(
    "Objects with confidence below the selected "
    "threshold will not be displayed."
)


# =========================================================
# IMAGE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)


# =========================================================
# MAIN DETECTION
# =========================================================

if uploaded_file is not None:

    # -----------------------------------------------------
    # READ IMAGE
    # -----------------------------------------------------

    pil_image = Image.open(uploaded_file).convert("RGB")

    image_rgb = np.array(pil_image)

    # Convert RGB → BGR for OpenCV
    image = cv2.cvtColor(
        image_rgb,
        cv2.COLOR_RGB2BGR
    )

    h, w = image.shape[:2]


    # -----------------------------------------------------
    # CREATE BLOB
    # -----------------------------------------------------

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )


    # -----------------------------------------------------
    # RUN MODEL
    # -----------------------------------------------------

    net.setInput(blob)

    detections = net.forward()


    # -----------------------------------------------------
    # CREATE OUTPUT IMAGE
    # -----------------------------------------------------

    output_image = image.copy()

    detected_objects = []
    confidence_scores = []


    # -----------------------------------------------------
    # PROCESS DETECTIONS
    # -----------------------------------------------------

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence >= confidence_threshold:

            class_id = int(
                detections[0, 0, i, 1]
            )

            label = CLASSES[class_id]

            detected_objects.append(label)

            confidence_scores.append(confidence)


            # -------------------------------------------------
            # BOUNDING BOX
            # -------------------------------------------------

            box = (
                detections[0, 0, i, 3:7]
                * np.array([w, h, w, h])
            )

            startX, startY, endX, endY = box.astype("int")


            # Keep coordinates within image
            startX = max(0, startX)
            startY = max(0, startY)

            endX = min(w - 1, endX)
            endY = min(h - 1, endY)


            # -------------------------------------------------
            # CLASS COLOR
            # -------------------------------------------------

            color = CLASS_COLORS.get(
                label,
                DEFAULT_COLOR
            )


            # -------------------------------------------------
            # DRAW BOUNDING BOX
            # -------------------------------------------------

            cv2.rectangle(
                output_image,
                (startX, startY),
                (endX, endY),
                color,
                3
            )


            # -------------------------------------------------
            # LABEL
            # -------------------------------------------------

            text = (
                f"{label}: "
                f"{confidence * 100:.1f}%"
            )

            # Text background
            (text_width, text_height), baseline = (
                cv2.getTextSize(
                    text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )
            )

            text_y = max(startY - 10, text_height + 10)

            cv2.rectangle(
                output_image,
                (
                    startX,
                    text_y - text_height - baseline
                ),
                (
                    startX + text_width,
                    text_y + baseline
                ),
                color,
                -1
            )


            # White text
            cv2.putText(
                output_image,
                text,
                (startX, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )


    # -----------------------------------------------------
    # CONVERT OUTPUT BGR → RGB
    # -----------------------------------------------------

    output_rgb = cv2.cvtColor(
        output_image,
        cv2.COLOR_BGR2RGB
    )


    # =====================================================
    # DISPLAY IMAGES
    # =====================================================

    st.subheader("🖼️ Detection Result")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Original Image")

        st.image(
            pil_image,
            use_container_width=True
        )

    with col2:

        st.markdown("### Detected Objects")

        st.image(
            output_rgb,
            use_container_width=True
        )


    # =====================================================
    # RESULTS
    # =====================================================

    st.divider()

    st.subheader("📊 Detection Summary")


    if detected_objects:

        object_counts = Counter(
            detected_objects
        )


        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

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


        # -------------------------------------------------
        # DETECTED CLASSES
        # -------------------------------------------------

        st.markdown("### 🔎 Detected Classes")

        for label, count in object_counts.items():

            st.write(
                f"**{label.capitalize()}** — {count}"
            )


        # -------------------------------------------------
        # CONFIDENCE DETAILS
        # -------------------------------------------------

        st.markdown("### 📈 Confidence Details")

        confidence_col1, confidence_col2, confidence_col3 = (
            st.columns(3)
        )

        with confidence_col1:

            st.write(
                "Lowest Confidence"
            )

            st.write(
                f"### {min(confidence_scores) * 100:.2f}%"
            )

        with confidence_col2:

            st.write(
                "Average Confidence"
            )

            st.write(
                f"### {np.mean(confidence_scores) * 100:.2f}%"
            )

        with confidence_col3:

            st.write(
                "Highest Confidence"
            )

            st.write(
                f"### {max(confidence_scores) * 100:.2f}%"
            )


        # -------------------------------------------------
        # DOWNLOAD RESULT
        # -------------------------------------------------

        st.divider()

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
            "No objects were detected above the "
            "selected confidence threshold."
        )
