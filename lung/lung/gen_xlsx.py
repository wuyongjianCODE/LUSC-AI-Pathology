import os
import json
import numpy as np
import pandas as pd
import skimage.io
from mrcnn.config import Config
from mrcnn import model as modellib
from mrcnn import utils
from tqdm import tqdm
import time

# Configuration for the lung segmentation model
class LungConfig(Config):
    NAME = "lung"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 2  # Background + tumor + stroma
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
    TRAIN_ROIS_PER_IMAGE = 32
    STEPS_PER_EPOCH = 1000
    VALIDATION_STEPS = 50


class InferenceConfig(LungConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def train_lung_segmentation():
    # Load training dataset
    dataset_train = LungDataset()
    dataset_train.load_lung("/data1/wyj/M/datasets/LUNG/")
    dataset_train.prepare()

    # Load validation dataset
    dataset_val = LungDataset()
    dataset_val.load_lung("/data1/wyj/M/datasets/LUNG/")
    dataset_val.prepare()

    # Create model in training mode
    config = LungConfig()
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir="/data1/wyj/M/logs/")

    # # Load pre-trained weights
    # weights_path = 'best.h5'
    # model.load_weights(weights_path, by_name=True, exclude=[
    #     "mrcnn_class_logits", "mrcnn_bbox_fc",
    #     "mrcnn_bbox", "mrcnn_mask"])

    # Train the model
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')

    return model


class LungDataset(utils.Dataset):
    def load_lung(self, dataset_dir, subset=None):
        # Add classes
        self.add_class("lung", 1, "tumor")
        self.add_class("lung", 2, "stroma")

        # Load annotations
        annotations = json.load(open(os.path.join(dataset_dir, "annotations.json")))
        images = annotations['images']
        annotations = annotations['annotations']

        # Add images
        for image in images:
            image_id = image['id']
            width = image['width']
            height = image['height']
            filename = image['file_name']

            self.add_image(
                "lung",
                image_id=image_id,
                path=os.path.join(dataset_dir, "images", filename),
                width=width, height=height)

    def load_mask(self, image_id):
        # Get annotations for this image
        annotations = [a for a in self.image_info[image_id]['annotations']
                       if a['image_id'] == image_id]

        # Create one mask per class
        count = len(annotations)
        masks = np.zeros([self.image_info[image_id]['height'],
                          self.image_info[image_id]['width'], count],
                         dtype=np.uint8)

        classes = []
        for i, a in enumerate(annotations):
            masks[:, :, i] = a['mask']
            classes.append(a['category_id'])

        # Handle cases where there are no annotations
        if count == 0:
            return np.zeros([self.image_info[image_id]['height'],
                             self.image_info[image_id]['width'], 0],
                            dtype=np.uint8), np.zeros((0,), dtype=np.int32)

        return masks, np.array(classes, dtype=np.int32)


def generate_density_report(density_file, json_dir, output_dir, max_retries=5, retry_delay=1):
    density_df = pd.concat([
        pd.read_excel(density_file, sheet_name='北京病例'),
        pd.read_excel(density_file, sheet_name='深圳病例')
    ])

    results = []
    for filename in density_df['filename']:
        json_name = f"{os.path.splitext(filename)[0]}.json"
        json_path = os.path.join(json_dir, json_name)

        retry_count = 0
        while True:
            try:
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    target = density_df[density_df['filename'] == filename]['STROMA_DENSITY_IN_RANGE_500um'].values[0]
                    actual = calculate_density(data)
                    results.append({
                        'filename': filename,
                        'target_density': target,
                        'calculated_density': actual,
                        'difference_pct': abs(target - actual) / target * 100 if target != 0 else 0
                    })
                break

            except (json.JSONDecodeError, IOError, KeyError, IndexError) as e:
                retry_count += 1
                if max_retries is not None and retry_count >= max_retries:
                    break

                time.sleep(retry_delay)

    result_df = pd.DataFrame(results)
    print("\nDensity Report:")
    print(result_df)
    print(f"\nAverage difference: {result_df['difference_pct'].mean():.2f}%")

    report_path = os.path.join(output_dir, '预测后的细胞密度结果.xlsx')
    result_df.to_excel(report_path, index=False)
    print(f"\nReport saved to: {report_path}")
    return result_df

def predict_on_classification_data(model, output_dir):
    # Create inference config
    inference_config = InferenceConfig()

    # Create model in inference mode
    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=output_dir)

    # Load trained weights
    model_path = os.path.join("/data1/wyj/M/logs/", "mask_rcnn_lung.h5")
    model.load_weights(model_path, by_name=True)

    # Get filenames from Excel
    density_df = pd.concat([
        pd.read_excel("/data1/wyj/M/预测后的细胞密度结果.xlsx", sheet_name='北京病例'),
        pd.read_excel("/data1/wyj/M/预测后的细胞密度结果.xlsx", sheet_name='深圳病例')
    ])
    filenames = density_df['filename'].unique()

    # Process each filename group
    for filename in tqdm(filenames, desc="Processing files"):
        # Find all images with this filename prefix
        image_dir =r'/data3/深圳分院35例/'# "/data3/beijingnew/山西肿瘤-新切新染_svs/" #r'/data3/深圳分院35例/' #'/data3/beijingnew/2-新辅助鳞癌术后LN评估-130例-359张-20240227_svs/'
        image_files = [f for f in os.listdir(image_dir)
                       if f.startswith(filename)]

        results = {
            "filename": filename,
            "images": [],
            "annotations": []
        }

        annotation_id = 1

        for image_file in image_files:
            # Load image
            image_path = os.path.join(image_dir, image_file)
            image = skimage.io.imread(image_path)

            # Detect objects
            r = model.detect([image], verbose=0)[0]

            # Add image info
            image_info = {
                "file_name": image_file,
                "width": image.shape[1],
                "height": image.shape[0],
                "id": len(results["images"]) + 1
            }
            results["images"].append(image_info)

            # Add annotations
            for i in range(len(r['rois'])):
                y1, x1, y2, x2 = r['rois'][i]
                class_id = r['class_ids'][i]

                annotation = {
                    "id": annotation_id,
                    "image_id": image_info["id"],
                    "category_id": int(class_id),
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "area": int((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0
                }
                results["annotations"].append(annotation)
                annotation_id += 1

        # Save results for this filename group
        output_path = os.path.join(output_dir, f"{filename}.json")
        with open(output_path, 'w') as f:
            json.dump(results, f)
def calculate_density(json_path):
    with open(json_path) as f:
        data = json.load(f)
    import math
    central_area = math.pi * (500 * 17.4 / 2) ** 2
    central_anns = 0

    # Find central patch (middle image)
    if len(data['images']) > 0:
        central_patch_id = len(data['images']) // 2 + 1

        for ann in data['annotations']:
            if ann['image_id'] == central_patch_id:
                x, y = ann['bbox'][0] + ann['bbox'][2] / 2, ann['bbox'][1] + ann['bbox'][3] / 2
                if math.sqrt((x - 112) ** 2 + (y - 112) ** 2) <= 500 * 17.4 / 2:
                    central_anns += 1

        return central_anns / central_area if central_area > 0 else 0

if __name__ == "__main__":

    density_file = "/data1/wyj/M/预测后的细胞密度结果.xlsx"
    json_dir = "/data1/wyj/M/WSI_json/"
    generate_density_report(density_file, json_dir, '/data1/wyj/M')