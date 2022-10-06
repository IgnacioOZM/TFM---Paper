for folder in normal_path:
    # Obtain all files inside each folder
    image, normals, _, instance_segmap = data_loading(normal_path, folder)
    image_aruco, _, _, _ = data_loading(aruco_path, folder)

    # Load and treatment of Coco Annotations
    bboxes, _ = coco_treatment(normal_path, folder)

    # Analyze and detect indexes of interest
    id_category = PIECE
    indexes = [index in categories if id_category]

    # Analysis of each piece
    for _, j in enumerate(indexes):
        # Creation of crops and category
        bbox = bboxes[j]
        image_crop = crop_creation(image, bbox)
        image_aruco_crop = crop_creation(image_aruco, bbox)
        image_normals_crop = crop_creation(normals, bbox)

        # Aruco detection
        corners, ids, Point_names = detect_markers(image_aruco_crop,
                                                    matrix_flag=True, 
                                                    distortion_flag=True)

        # Exclusion of points from other pieces
        corners_clean, ids_clean, Point_names_clean = segmentation_detect(
            bbox, corners, total_padding, instance_segmap)

        # Extraction of normals
        normal_coords = normal_extraction(
            image_normals_crop, corners_clean)

        # Saving of images and data
        data_saving(num_images, save_path, image_crop,
                    corners_clean, Point_names_clean)

        # Increase the number of processed images
        num_images += 1
