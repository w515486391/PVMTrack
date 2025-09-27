class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/yy/4T/wbb4/pvmtrack-main'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/media/yy/4T/wbb4/pvmtrack-main/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/media/yy/4T/wbb4/pvmtrack-main/pretrained_networks'
        self.lasot_dir = '/media/yy/2T/wbb2/dataset/lasot'
        self.got10k_dir = '/media/yy/ab96238f-21c5-4b6b-b12b-dfc7fc9ae4ec/media/wbb3/dataset/GOT10k/train'
        self.got10k_val_dir = '/media/yy/ab96238f-21c5-4b6b-b12b-dfc7fc9ae4ec/media/wbb3/dataset/GOT10k/val'
        self.lasot_lmdb_dir = '/media/yy/4T/wbb4/pvmtrack-main/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/media/yy/4T/wbb4/pvmtrack-main/data/got10k_lmdb'
        self.trackingnet_dir = '/media/yy/2T/wbb2/dataset/TrackingNet'
        self.trackingnet_lmdb_dir = '/media/yy/4T/wbb4/pvmtrack-main/data/trackingnet_lmdb'
        self.coco_dir = '/media/yy/2T/wbb2/dataset/coco'
        self.coco_lmdb_dir = '/media/yy/4T/wbb4/pvmtrack-main/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/media/yy/4T/wbb4/pvmtrack-main/data/vid'
        self.imagenet_lmdb_dir = '/media/yy/4T/wbb4/pvmtrack-main/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
