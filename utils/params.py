import torch


class Parameters:
    def __init__(self):

        self.weights = "best.pt"

        self.imgsz = 640
        self.conf_thres = 0.25
        self.max_det = 1000
        self.hide_conf = True

        self.region_threshold = 0.05

        self.color_blue = (255, 255, 0)
        self.color_red = (25, 20, 240)
        self.color = self.color_blue
        self.text_x_align = 10
        self.inference_time_y = 30
        self.fps_y = 90
        self.analysis_time_y = 60
        self.font_scale = 0.7
        self.thickness = 2
        self.rect_thickness = 3

        self.rect_size = 15000

        self.pred_shape = (480, 640, 3)
        self.vis_shape = (800, 600)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = "/home/mef/Documents/plate_detection_project/best.pt"
