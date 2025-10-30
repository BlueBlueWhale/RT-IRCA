from ultralytics.models.yolo.model import YOLO
from ultralytics.models.rt_irca.train import RTIRCATrainer

# Load the student model
student = YOLO("yolo11n.pt")
print("Student model loaded")

# Perform MLKD training
results = student.train(trainer=RTIRCATrainer, cfg="mlkd.yaml")
