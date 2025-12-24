from ultralytics import YOLO


def predict_image(img_path: str = "sample/OIP.jpg"):
    model = YOLO(model="experiments/exp001/logs/train/weights/best.pt")  # Sample model
    results = model.predict(img_path)

    labels = model.names
    preds = []

    # Get label, conf and box of each detection
    for i, result in enumerate(results):
        boxes = result.boxes.xyxy
        classes = result.boxes.cls.reshape(-1, 1)
        confs = result.boxes.conf.reshape(-1, 1)

        for j in range(boxes.shape[0]):
            preds.append(
                (
                    boxes[j],
                    classes[j],
                    confs[j],
                )
            )

    # Sort by x coordinate to print the number in order
    preds.sort(key=lambda x: x[0][0])
    for i in preds:
        print(labels[i[1].item()], end="")
    print()

    # Also show the image
    results[0].show()


if __name__ == "__main__":
    predict_image()
