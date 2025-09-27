import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class OTBDataset(BaseDataset):
    """ OTB-2015 dataset
    Publication:
        Object Tracking Benchmark
        Wu, Yi, Jongwoo Lim, and Ming-hsuan Yan
        TPAMI, 2015
        http://faculty.ucmerced.edu/mhyang/papers/pami15_tracking_benchmark.pdf
    Download the dataset from http://cvlab.hanyang.ac.kr/tracker_benchmark/index.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.otb_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path, 
        sequence_path=sequence_path, frame=frame_num, nz=nz, ext=ext) for frame_num in range(start_frame+init_omit, end_frame+1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'otb', ground_truth_rect[init_omit:,:],
                        object_class=sequence_info['object_class'])

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "S0310", "path": "seq/S0310", "startFrame": 1, "endFrame": 118, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0310_gt.txt", "object_class": "uav"},
            {"name": "S1501", "path": "seq/S1501", "startFrame": 1, "endFrame": 254, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1501_gt.txt", "object_class": "uav"},
            {"name": "S1312", "path": "seq/S1312", "startFrame": 1, "endFrame": 1919, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1312_gt.txt", "object_class": "uav"},
            {"name": "S1307", "path": "seq/S1307", "startFrame": 1, "endFrame": 742, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1307_gt.txt", "object_class": "uav"},
            {"name": "S1602", "path": "seq/S1602", "startFrame": 1, "endFrame": 838, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1602_gt.txt", "object_class": "uav"},
            {"name": "S1313", "path": "seq/S1313", "startFrame": 1, "endFrame": 2045, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1313_gt.txt", "object_class": "uav"},
            {"name": "S0402", "path": "seq/S0402", "startFrame": 1, "endFrame": 561, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0402_gt.txt", "object_class": "uav"},
            {"name": "S0103", "path": "seq/S0103", "startFrame": 1, "endFrame": 1135, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0103_gt.txt", "object_class": "uav"},
            {"name": "S1701", "path": "seq/S1701", "startFrame": 1, "endFrame": 324, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1701_gt.txt", "object_class": "uav"},
            {"name": "S0801", "path": "seq/S0801", "startFrame": 1, "endFrame": 526, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0801_gt.txt", "object_class": "uav"},
            {"name": "S0303", "path": "seq/S0303", "startFrame": 1, "endFrame": 200, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0303_gt.txt", "object_class": "uav"},
            {"name": "S1301", "path": "seq/S1301", "startFrame": 1, "endFrame": 537, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1301_gt.txt", "object_class": "uav"},
            {"name": "S1311", "path": "seq/S1311", "startFrame": 1, "endFrame": 456, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1311_gt.txt", "object_class": "uav"},
            {"name": "S1607", "path": "seq/S1607", "startFrame": 1, "endFrame": 563, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1607_gt.txt", "object_class": "uav"},
            {"name": "S1306", "path": "seq/S1306", "startFrame": 1, "endFrame": 2435, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1306_gt.txt", "object_class": "uav"},
            {"name": "S0306", "path": "seq/S0306", "startFrame": 1, "endFrame": 295, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0306_gt.txt", "object_class": "uav"},
            {"name": "S1310", "path": "seq/S1310", "startFrame": 1, "endFrame": 805, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1310_gt.txt", "object_class": "uav"},
            {"name": "S0201", "path": "seq/S0201", "startFrame": 1, "endFrame": 948, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0201_gt.txt", "object_class": "uav"},
            {"name": "S1605", "path": "seq/S1605", "startFrame": 1, "endFrame": 605, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1605_gt.txt", "object_class": "uav"},
            {"name": "S1702", "path": "seq/S1702", "startFrame": 1, "endFrame": 329, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1702_gt.txt", "object_class": "uav"},
            {"name": "S0302", "path": "seq/S0302", "startFrame": 1, "endFrame": 440, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0302_gt.txt", "object_class": "uav"},
            {"name": "S0301", "path": "seq/S0301", "startFrame": 1, "endFrame": 695, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0301_gt.txt", "object_class": "uav"},
            {"name": "S1202", "path": "seq/S1202", "startFrame": 1, "endFrame": 329, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1202_gt.txt", "object_class": "uav"},
            {"name": "S1309", "path": "seq/S1309", "startFrame": 1, "endFrame": 1302, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1309_gt.txt", "object_class": "uav"},
            {"name": "S1308", "path": "seq/S1308", "startFrame": 1, "endFrame": 983, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1308_gt.txt", "object_class": "uav"},
            {"name": "S0307", "path": "seq/S0307", "startFrame": 1, "endFrame": 414, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0307_gt.txt", "object_class": "uav"},
            {"name": "S1101", "path": "seq/S1101", "startFrame": 1, "endFrame": 298, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1101_gt.txt", "object_class": "uav"},
            {"name": "S1201", "path": "seq/S1201", "startFrame": 1, "endFrame": 2534, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1201_gt.txt", "object_class": "uav"},
            {"name": "S1603", "path": "seq/S1603", "startFrame": 1, "endFrame": 2969, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1603_gt.txt", "object_class": "uav"},
            {"name": "S0401", "path": "seq/S0401", "startFrame": 1, "endFrame": 501, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0401_gt.txt", "object_class": "uav"},
            {"name": "S1304", "path": "seq/S1304", "startFrame": 1, "endFrame": 519, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1304_gt.txt", "object_class": "uav"},
            {"name": "S1604", "path": "seq/S1604", "startFrame": 1, "endFrame": 624, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1604_gt.txt", "object_class": "uav"},
            {"name": "S1601", "path": "seq/S1601", "startFrame": 1, "endFrame": 468, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1601_gt.txt", "object_class": "uav"},
            {"name": "S1303", "path": "seq/S1303", "startFrame": 1, "endFrame": 1112, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1303_gt.txt", "object_class": "uav"},
            {"name": "S1401", "path": "seq/S1401", "startFrame": 1, "endFrame": 188, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1401_gt.txt", "object_class": "uav"},
            {"name": "S0501", "path": "seq/S0501", "startFrame": 1, "endFrame": 232, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0501_gt.txt", "object_class": "uav"},
            {"name": "S1302", "path": "seq/S1302", "startFrame": 1, "endFrame": 403, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1302_gt.txt", "object_class": "uav"},
            {"name": "S0101", "path": "seq/S0101", "startFrame": 1, "endFrame": 1784, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0101_gt.txt", "object_class": "uav"},
            {"name": "S0601", "path": "seq/S0601", "startFrame": 1, "endFrame": 82, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0601_gt.txt", "object_class": "uav"},
            {"name": "S0701", "path": "seq/S0701", "startFrame": 1, "endFrame": 596, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0701_gt.txt", "object_class": "uav"},
            {"name": "S1305", "path": "seq/S1305", "startFrame": 1, "endFrame": 1378, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1305_gt.txt", "object_class": "uav"},
            {"name": "S0308", "path": "seq/S0308", "startFrame": 1, "endFrame": 319, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0308_gt.txt", "object_class": "uav"},
            {"name": "S0102", "path": "seq/S0102", "startFrame": 1, "endFrame": 350, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0102_gt.txt", "object_class": "uav"},
            {"name": "S0901", "path": "seq/S0901", "startFrame": 1, "endFrame": 350, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0901_gt.txt", "object_class": "uav"},
            {"name": "S0304", "path": "seq/S0304", "startFrame": 1, "endFrame": 359, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0304_gt.txt", "object_class": "uav"},
            {"name": "S1606", "path": "seq/S1606", "startFrame": 1, "endFrame": 655, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1606_gt.txt", "object_class": "uav"},
            {"name": "S0305", "path": "seq/S0305", "startFrame": 1, "endFrame": 706, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0305_gt.txt", "object_class": "uav"},
            {"name": "S0309", "path": "seq/S0309", "startFrame": 1, "endFrame": 214, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0309_gt.txt", "object_class": "uav"},
            {"name": "S0602", "path": "seq/S0602", "startFrame": 1, "endFrame": 292, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S0602_gt.txt", "object_class": "uav"},
            {"name": "S1001", "path": "seq/S1001", "startFrame": 1, "endFrame": 353, "nz": 6, "ext": "jpg",
             "anno_path": "anno/S1001_gt.txt", "object_class": "uav"}
        ]
    
        return sequence_info_list