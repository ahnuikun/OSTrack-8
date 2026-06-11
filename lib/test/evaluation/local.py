from lib.test.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    data_dir = '/home/professor4/OSTrack-main/data'
    save_dir = '/home/professor4/OSTrack-main/output'

    settings.prj_dir = '/home/professor4/OSTrack-main'
    settings.save_dir = save_dir
    settings.results_path = save_dir + '/test/tracking_results'
    settings.segmentation_path = save_dir + '/test/segmentation_results'
    settings.network_path = save_dir + '/test/networks'
    settings.result_plot_path = save_dir + '/test/result_plots'
    settings.otb_path = data_dir + '/otb'
    settings.nfs_path = data_dir + '/nfs'
    settings.uav_path = data_dir + '/uav123'
    settings.visdrone_path = data_dir + '/visdrone'
    settings.uavdt_path = data_dir + '/uavdt'
    settings.dtb70_path = data_dir + '/DTB70'
    settings.tc128_path = data_dir + '/TC128'
    settings.tpl_path = ''
    settings.vot_path = data_dir + '/VOT2019'
    settings.vot18_path = data_dir + '/vot2018'
    settings.vot22_path = data_dir + '/vot2022'
    settings.got10k_path = data_dir + '/got10k'
    settings.got10k_lmdb_path = data_dir + '/got10k_lmdb'
    settings.lasot_path = data_dir + '/lasot'
    settings.lasot_lmdb_path = data_dir + '/lasot_lmdb'
    settings.trackingnet_path = data_dir + '/trackingnet'
    settings.itb_path = data_dir + '/itb'
    settings.tnl2k_path = data_dir + '/tnl2k'
    settings.lasot_extension_subset_path_path = data_dir + '/lasot_extension_subset'
    settings.davis_dir = ''
    settings.youtubevos_dir = ''

    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.tn_packed_results_path = ''

    return settings
