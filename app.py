import streamlit as st
import cv2
import numpy as np
from PIL import Image
from collections import Counter


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="MobileNet-SSD Object Detection",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# CLASS LABELS
# --------------------------------------------------

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


# --------------------------------------------------
# CLASS COLORS
# OpenCV uses BGR format
# --------------------------------------------------

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
    "bird": (255, 128, 0)
}

DEFAULT_COLOR = (255, 255, 255)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

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


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🤖 MobileNet-SSD Object Detection")

st.write(
    "Upload an image to detect objects using a pretrained "
    "MobileNet-SSD model."
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Detection Settings")

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.50,
    max_value=1.00,
    value=0.50,
    step=0.05
)

st.sidebar.write(
    f"Current threshold: **{confidence_threshold * 100:.0f}%**"
)


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload an image",
    type=["jpg", "jpeg", "png"]
)


# --------------------------------------------------
# OBJECT DETECTION
# --------------------------------------------------

if uploaded_file is not None:

    # Read uploaded image
    pil_image = Image.open(uploaded_file).convert("RGB")

    image = np.array(pil_image)

    # Convert RGB to BGR for OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w = image.shape[:2]

    # --------------------------------------------------
    # CREATE BLOB
    # --------------------------------------------------

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    # --------------------------------------------------
    # RUN DETECTION
    # --------------------------------------------------

    net.setInput(blob)

    detections = net.forward()

    # Copy original image
    output_image = image.copy()

    detected_objects = []
    confidence_scores = []

    # --------------------------------------------------
    # PROCESS DETECTIONS
    # --------------------------------------------------

    for i in range(detections.shape[2]):

        confidence = detections[0, 0, i, 2]

        if confidence >= confidence_threshold:

            class_id = int(detections[0, 0, i, 1])

            label = CLASSES[class_id]

            confidence_scores.append(confidence)

            detected_objects.append(label)

            # Calculate bounding box
            box = (
                detections[0, 0, i, 3:7]
                * np.array([w, h, w, h])
            )

            startX, startY, endX, endY = box.astype("int")

            # Keep coordinates inside image
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(w - 1, endX)
            endY = min(h - 1, endY)

            # Get class-specific color
            color = CLASS_COLORS.get(
                label,
                DEFAULT_COLOR
            )

            # Draw bounding box
            cv2.rectangle(
                output_image,
                (startX, startY),
                (endX, endY),
                color,
                2
            )

            # Create label
            text = f"{label}: {confidence * 100:.1f}%"

            cv2.putText(
                output_image,
                text,
                (startX, max(startY - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    # Convert output back to RGB
    output_rgb = cv2.cvtColor(
        output_image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------
    # DISPLAY IMAGES
    # --------------------------------------------------

    st.subheader("Detection Result")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Original Image")

        st.image(
            pil_image,
            use_container_width=True
        )

    with col2:

        st.write("### Detected Objects")

        st.image(
            output_rgb,
            use_container_width=True
        )


    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    st.subheader("📊 Detection Summary")

    if detected_objects:

        object_counts = Counter(detected_objects)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Objects",
                len(detected_objects)
            )

        with col2:
            st.metric(
                "Average Confidence",
                f"{np.mean(confidence_scores) * 100:.2f}%"
            )

        with col3:
            st.metric(
                "Highest Confidence",
                f"{max(confidence_scores) * 100:.2f}%"
            )


        st.write("### Detected Classes")

        for label, count in object_counts.items():

            st.write(
                f"**{label.capitalize()}**: {count}"
            )


        # --------------------------------------------------
        # CONFIDENCE DETAILS
        # --------------------------------------------------

        st.write("### Confidence Details")

        st.write(
            f"Lowest confidence: "
            f"**{min(confidence_scores) * 100:.2f}%**"
        )

        st.write(
            f"Highest confidence: "
            f"**{max(confidence_scores) * 100:.2f}%**"
        )

        st.write(
            f"Average confidence: "
            f"**{np.mean(confidence_scores) * 100:.2f}%**"
        )


        # --------------------------------------------------
        # DOWNLOAD RESULT
        # --------------------------------------------------

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
