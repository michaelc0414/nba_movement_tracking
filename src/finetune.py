from ultralytics import YOLO


if __name__ == "__main__":    
    #starting model
    model = YOLO("models/yolo26n.pt")

    # search_space = {
    #     "lr0": (1e-5, 1e-2),
    #     "weight_decay": (0.0, 0.001),
    #     "lrf": (0.01, 1.0),
    #     "box": (1.0, 20.0),
    #     "cls": (0.1, 4.0),
    #     "dfl": (0.4, 12.0),
    #     "hsv_h": (0.0, 0.1),
    #     "hsv_s": (0.0, 0.9),
    #     "hsv_v": (0.0, 0.9),
    #     "translate": (0.0, 0.9),
    #     "scale": (0.0, 0.95),
    #     "fliplr": (0.0, 1.0),
    #     "mosaic": (0.0, 1.0),

    # }

    results = model.train(
        data="data/tuning/data.yaml",
        epochs=50,
        imgsz=960,
        batch=4,
        patience=10,

        device=0,
        project="models",
        name="player_detector_polished",
        exist_ok=True,
        pretrained=True,

        optimizer="AdamW",
        lr0=0.00411, #from search space h-params
        weight_decay=0.0, #from search space h-params
        lrf=0.01133, #from search space h-params
        momentum=0.73375,
        warmup_epochs=1.22935,
        warmup_momentum=0.1525,
        box=6.75945,
        cls=0.50434,
        dfl=1.62141,
        hsv_h=0.02756,
        hsv_s=0.80488,
        hsv_v=0.31733, #from search space h-params
        degrees=0.0,
        translate=0.15631, #from search space h-params
        scale=0.35857, #from search space h-params
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.63646, #from search space h-params
        mosaic=0.92013, #from search space h-params
        mixup=0.0,
        copy_paste=0.0,

        # space=search_space,
        # iterations=30,
        # augment=True,
        val=True,
        save=True,
        plots=True,
        save_period=10,
        workers=2,
        cache='disk'
    )

    print("Model trained")