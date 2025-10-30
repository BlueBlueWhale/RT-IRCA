from ultralytics.nn.tasks import DetectionModel
from .loss import MLKDLoss


class RTIRCA(DetectionModel):
    """RT-IRCA (Real-time Infrared Context Aggregation) for Substation Equipment Detection"""

    def __init__(
        self,
        cfg="yolo11s.yaml",
        ch=3,
        nc=None,
        verbose=True,
        alpha=1e-3,
        beta=1e-8,
        gamma=1e-6,
        temperature=0.5,
        has_rev=True,
        has_irca=True,
        has_mut=True,
        teacher=None,
        layer_indices=[13, 16, 19, 22],
        student_channels=None,
        teacher_channels=None,
    ):
        """Initialize the RTIRCA model."""
        super().__init__(cfg, nc=nc, ch=ch, verbose=verbose)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.temperature = temperature
        self.has_rev = has_rev
        self.has_irca = has_irca
        self.has_mut = has_mut
        self.teacher = teacher
        self.layer_indices = layer_indices
        self.student_channels = student_channels
        self.teacher_channels = teacher_channels

    def init_criterion(self):
        """Initialize the loss criterion for the model."""
        return MLKDLoss(self)
