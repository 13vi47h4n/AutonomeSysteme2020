from Ultralight.vision.ssd.config.fd_config import define_img_size

class FastFaceRecognition:

    def __init__(self):
        test_device = "cpu"
        self.candidate_size = 1000 # args.candidate_size
        self.threshold = 0.7 # args.threshold

        label_path = "./models/voc-model-labels.txt"
        net_type = "RFB"

        class_names = [name.strip() for name in open(label_path).readlines()]
        num_classes = len(class_names)

        input_img_size = 480
        define_img_size(input_img_size)

        from Ultralight.vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
        from Ultralight.vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
        from Ultralight.vision.utils.misc import Timer

        if net_type == 'slim':
            model_path = "models/pretrained/version-slim-320.pth"
            # model_path = "models/pretrained/version-slim-640.pth"
            net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
            self.predictor = create_mb_tiny_fd_predictor(net, candidate_size=self.candidate_size, device=test_device)
        elif net_type == 'RFB':
            model_path = "models/pretrained/version-RFB-320.pth"
            # model_path = "models/pretrained/version-RFB-640.pth"
            net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
            self.predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=self.candidate_size, device=test_device)
        else:
            print("The net type is wrong!")
            sys.exit(1)
        
        net.load(model_path)
    
    def face_locations(self, image):
        face_locations, _, _ = self.predictor.predict(image, self.candidate_size / 2, self.threshold)
        return face_locations[0]