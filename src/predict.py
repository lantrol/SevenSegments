from ultralytics import YOLO


def predict_image(img_path: str = "sample/OIP.jpg", im_show=False):
    model = YOLO(model="models/best.onnx", task="detect")  # Sample model
    results = model.predict(img_path)

    labels = model.names
    preds = []

    # Get label, conf and box of each detection
    for i, result in enumerate(results):
        boxes = result.boxes.xyxy
        classes = result.boxes.cls.reshape(-1, 1)
        confs = result.boxes.conf.reshape(-1, 1)

        for j in range(boxes.shape[0]):
            if confs[j] < 0.3:
                continue

            preds.append(
                (
                    boxes[j],
                    classes[j],
                    confs[j],
                )
            )

    # Sort by x coordinate to print the number in order
    print("Predicted reading: ")
    preds.sort(key=lambda x: x[0][0])
    for i in preds:
        print(labels[i[1].item()], end="")
    print()

    # Also show the image
    if im_show:
        results[0].show()


if __name__ == "__main__":
    predict_image("sample/counter.png", im_show=True)
