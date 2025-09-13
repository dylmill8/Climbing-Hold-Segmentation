import os 
from ultralytics import YOLO

def label_image(image_path="ml/model/test.jpg", model_path="ml/model/runs/y11m_seg_1024/weights/best.pt") -> None:
    """
    Labels an image using the current best model and saves the result.

    @param image: Path to the image file
    @param model: Path to the YOLO model weights
    """
    # Load Model
    model = YOLO(model_path)
    
    results = model(image_path)
    
    output_dir = "data/images/inference"
    os.makedirs(output_dir, exist_ok=True)
    
    
    for result in results:

      result.save(filename=os.path.join(output_dir, "test.jpg"))
      # result.show() # used for testing purposes

label_image()