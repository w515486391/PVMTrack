from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/media/yy/4T/wbb4/pvmtrack-main/data/got10k_lmdb'
    settings.got10k_path = '/media/yy/4T/wbb4/pvmtrack-main/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.itb_path = '/media/yy/4T/wbb4/pvmtrack-main/data/itb'
    settings.lasot_extension_subset_path_path = '/media/yy/4T/wbb4/pvmtrack-main/data/lasot_extension_subset'
    settings.lasot_lmdb_path = '/media/yy/4T/wbb4/pvmtrack-main/data/lasot_lmdb'
    settings.lasot_path = '/media/yy/4T/wbb4/pvmtrack-main/data/lasot'
    settings.network_path = '/media/yy/4T/wbb4/pvmtrack-main/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = '/media/yy/4T/wbb4/pvmtrack-main/data/nfs'
    settings.otb_path = '/media/yy/2T/wbb2/dataset/UAVDT'
    # /home/yy/wbb1/1T/wbb1/dataset/OTB100
    # /media/yy/2T/wbb2/dataset/UAVDT
    # /media/yy/2T/wbb2/dataset/UAVTrack112
    settings.prj_dir = '/media/yy/4T/wbb4/pvmtrack-main'
    settings.result_plot_path = '/media/yy/4T/wbb4/pvmtrack-main/output/test/result_plots'
    settings.results_path = '/media/yy/4T/wbb4/pvmtrack-main/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/media/yy/4T/wbb4/pvmtrack-main/output'
    settings.segmentation_path = '/media/yy/4T/wbb4/pvmtrack-main/output/test/segmentation_results'
    settings.tc128_path = '/media/yy/4T/wbb4/pvmtrack-main/data/TC128'
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/media/yy/4T/wbb4/pvmtrack-main/data/tnl2k'
    settings.tpl_path = ''
    settings.trackingnet_path = '/media/yy/4T/wbb4/pvmtrack-main/data/trackingnet'

    settings.uav123_path = '/media/yy/4T/wbb4/wbb1/dataset/UAV123'
    settings.uav123_10fps_path = '/media/yy/2T/wbb2/dataset/UAV123_10fps'
    settings.uavdt_path = '/media/yy/2T/wbb2/dataset/UAVDT'
    settings.dtb70_path = '/media/yy/2T/wbb2/dataset/DTB70'
    settings.uav_path = '/media/yy/2T/wbb2/dataset/UAV20L'
    settings.visdrone_path = '/media/yy/4T/wbb4/wbb1/visdrone'

    # settings.uav_path = '/media/yy/4T/wbb4/pvmtrack-main/data/uav'
    settings.vot18_path = '/media/yy/4T/wbb4/pvmtrack-main/data/vot2018'
    settings.vot22_path = '/media/yy/4T/wbb4/pvmtrack-main/data/vot2022'
    settings.vot_path = '/media/yy/4T/wbb4/pvmtrack-main/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

