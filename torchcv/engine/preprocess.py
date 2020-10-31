from torchcv.preprocessing import ResizeImages, CreateCsv, CreateLabelMap, CalculateStats, CreateCsvWithAnnotations, \
    ResizeImagesAndAnnotations

PREPROCESS_ENGINE = {
    "resize": ResizeImages,
    "resize_with_annot": ResizeImagesAndAnnotations,
    "csv": CreateCsv,
    "csv_with_annot": CreateCsvWithAnnotations,
    "label_map": CreateLabelMap,
    "stats": CalculateStats
}
