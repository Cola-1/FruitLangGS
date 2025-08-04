# Baum 01


Baum_01_unet = {
    "path": "/FruitLangGS/tree01/tree01_pc.ply",
    "remove_outliers_nb_points": 120,
    "remove_outliers_radius": 0.015,
    "down_sample": 0.001,
    "eps": 0.02,
    "cluster_merge_distance": 0.04,
    "minimum_size_factor": 0.3,
    "min_samples": 100,
    "template_path": '/clustering/apple_template.ply',
    'apple_template_size': 1,
    'gt_cluster': None,
    'gt_count': 179,
}




# Fuji

Fuji_sam = {
    "path": "/FruitLangGS/data_pfuji01/pfuji01_pc.ply",
    "remove_outliers_radius": 0.03,
    "down_sample": 0.001,
    "cluster_merge_distance": 0.04,
    "minimum_size_factor": 0.2,
    "min_samples": 100,
    'template_path': './clustering/apple_template.ply',
    "remove_outliers_nb_points": 50,
    "eps": 0.02,
    "apple_template_size": 1,
    "gt_cluster": None,
    "gt_count": 1455

}

