from ultralytics import YOLO


if __name__ == "__main__":    
    #starting model
    model = YOLO("models/yolo26n.pt")

    search_space = {
        "lr0": (1e-5, 1e-2),
        "weight_decay": (0.0, 0.001),
        "lrf": (0.01, 1.0),
        "box": (1.0, 20.0),
        "cls": (0.1, 4.0),
        "dfl": (0.4, 12.0),
        "hsv_h": (0.0, 0.1),
        "hsv_s": (0.0, 0.9),
        "hsv_v": (0.0, 0.9),
        "translate": (0.0, 0.9),
        "scale": (0.0, 0.95),
        "fliplr": (0.0, 1.0),
        "mosaic": (0.0, 1.0),

    }

    results = model.tune(
        data="data/tuning/data.yaml",
        epochs=30,
        imgsz=640,
        batch=4,
        patience=10,

        device=0,
        project="models",
        name="player_detector",
        exist_ok=True,
        pretrained=True,

        optimizer="AdamW",
        # lr0=0.00269, search space h-params
        # weight_decay=0.00015, search space h-params
        # lrf=0.00288, search space h-params
        momentum=0.73375,
        warmup_epochs=1.22935,
        warmup_momentum=0.1525,
        # box=18.27875,
        # cls=1.32899,
        # dfl=0.56016,
        # hsv_h=0.01148,
        # hsv_s=0.53554,
        # hsv_v=0.13636, search space h-params
        degrees=0.0,
        # translate=0.12431, search space h-params
        # scale=0.07643, search space h-params
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        # fliplr=0.08631, search space h-params
        # mosaic=0.42551, search space h-params
        mixup=0.0,
        copy_paste=0.0,

        space=search_space,
        iterations=30,
        # augment=True,
        val=True,
        save=True,
        plots=True,
        save_period=10,
        workers=2,
        cache='disk'
    )

    print("Model trained")