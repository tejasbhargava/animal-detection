import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
new_model = YOLO("yolov8s.yaml")
def train_yolov8n():
    results = model.train(
        data="data.yaml",
        epochs=50,
        device=0,
        workers=0
    )
def train_yolov8s():
    new_model.train(
        data="data.yaml",
        epochs=75,
        imgsz=768,
        device=0,
        workers=0,
        name="yolov8smodel"
    )

def infer_yolov8n():
    my_model = YOLO("yolov8nmodel/weights/best.pt")
    result = my_model.predict(
        source="test/monke_test.jpg",
        conf=0.3,
        iou=0.45,
        device=0,
        save=True
    )
    # metrics = model.val(
    #     data="data.yaml",
    #     device=0,
    #     plots=True   
    # )
    annotated_img = result[0].plot()  # numpy array (BGR)

    # Convert BGR → RGB for matplotlib
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(annotated_img)
    plt.axis("off")
    plt.title("YOLOv8n Prediction")
    plt.show()


def infer_yolov8s():
    my_model = YOLO("yolov8smodel/weights/best.pt")
    result = my_model.predict(
        source="test/monke_test.jpg",
        conf=0.3,
        iou=0.45,
        device=0,
        save=True
    )
    # metrics = model.val(
    #     data="data.yaml",
    #     device=0,
    #     plots=True   
    # )
    annotated_img = result[0].plot()  # numpy array (BGR)

    # Convert BGR → RGB for matplotlib
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 8))
    plt.imshow(annotated_img)
    plt.axis("off")
    plt.title("YOLOv8s Prediction")
    plt.show()

def compare():
    model_n = YOLO("yolov8nmodel/weights/best.pt")
    model_s = YOLO("yolov8smodel/weights/best.pt")

    # Validate both models on same dataset
    metrics_n = model_n.val(data="data.yaml")
    metrics_s = model_s.val(data="data.yaml")

    # Create comparison table
    comparison = {
        "Model": ["YOLOv8n", "YOLOv8s"],
        "Precision (%)": [
            metrics_n.box.p * 100,
            metrics_s.box.p * 100
        ],
        "Recall (%)": [
            metrics_n.box.r * 100,
            metrics_s.box.r * 100
        ],
        "mAP@0.5 (%)": [
            metrics_n.box.map50 * 100,
            metrics_s.box.map50 * 100
        ],
        "mAP@0.5:0.95 (%)": [
            metrics_n.box.map * 100,
            metrics_s.box.map * 100
        ]
    }

    df = pd.DataFrame(comparison)
    print("\nModel Comparison:")
    print(df.round(2))

    
if __name__ == "__main__":
    # train() 
    #train_yolov8s()
    #infer_yolov8n()
    infer_yolov8s()
    #compare()