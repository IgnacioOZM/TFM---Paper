# Obtain everything inside input_paths
for folder in normal_path: 
    # Obtain all files inside each folder
    image = data_loading(normal_path, folder)

    # Load and treatment of Coco Annotations
    bboxes, categories = coco_treatment(normal_path, folder)

    # Saving of images and data
    data_saving(num_images, save_path, image, bboxes, categories)

    # Increase the number of processed images
    num_images += 1