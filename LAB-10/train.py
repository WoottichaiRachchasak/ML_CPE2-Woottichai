


from ultralytics import YOLO
from pathlib import Path

def main():

    # Load pretrained YOLO model
    model = YOLO("yolov8n.pt")

    # Train model
    model.train(
        data="VOC_small.yaml",
        #กำหนดให้ใช้ข้อมูลชุดเทรน 10% จากทั้งหมดเพื่อความรวดเร็ว
        fraction=0.05,
        epochs=15,
        imgsz=480,
        batch=8,
        #device = 0 คือใช้ gpu ประมาณผลเร็วกว่า device = cpu
        device="cpu",
        workers=0,
        amp=True,
        cache=False,
        project="runs",
        name="voc_e15",
        exist_ok=True,
        save=True,
        plots=True
    )

    # Output directory
    save_dir = Path(model.trainer.save_dir)

    print("Training completed!")
    print(f"Best Model: {save_dir / 'weights/best.pt'}")
    print(f"Last Model: {save_dir / 'weights/last.pt'}")
    print(f"Training CSV: {save_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
