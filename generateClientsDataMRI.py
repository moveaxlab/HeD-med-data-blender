import os
import shutil
import argparse
import random


def get_images_by_class(dataset_path):
    class_images = {}
    keyword_range = [150, 160]
    keyword_filter = [
        f"_{i}.jpg" for i in range(keyword_range[0], keyword_range[-1] + 1)
    ]

    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            images = [
                img
                for img in os.listdir(class_path)
                if os.path.isfile(os.path.join(class_path, img))
                and any(img.endswith(kw) for kw in keyword_filter)
            ]
            class_images[class_name] = images
    return class_images


def split_dataset(dataset_path, output_path, percentages):
    os.makedirs(output_path, exist_ok=True)
    class_images = get_images_by_class(dataset_path)
    distribution_counts = {f"subset_{i}": {} for i in range(len(percentages))}

    for i, percentage in enumerate(percentages):
        subset_path = os.path.join(output_path, f"subset_{i}")
        os.makedirs(subset_path, exist_ok=True)

        for class_name, images in class_images.items():
            class_subset_path = os.path.join(subset_path, class_name)
            os.makedirs(class_subset_path, exist_ok=True)
            num_samples = int(len(images) * percentage)
            selected_images = random.sample(images, num_samples)
            distribution_counts[f"subset_{i}"][class_name] = len(selected_images)

            for img in selected_images:
                src = os.path.join(dataset_path, class_name, img)
                dst = os.path.join(class_subset_path, img)
                shutil.copy2(src, dst)

    print("Image distribution across subsets:")
    for subset, classes in distribution_counts.items():
        print(f"{subset}:")
        for class_name, count in classes.items():
            print(f"  {class_name}: {count} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset into multiple subsets maintaining structure."
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        nargs="?",
        default="../../datasets/OASIS_MRI/Data",
        help="Path to the original dataset",
    )
    parser.add_argument(
        "output_path",
        type=str,
        nargs="?",
        default="../../datasets/OASIS_MRI_splitted",
        help="Path to save the subsets",
    )
    parser.add_argument(
        "percentages",
        type=float,
        nargs="*",
        default=[0.05, 0.55, 0.4],
        help="List of percentages for each subset",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        print("Error: Dataset path does not exist.")
        exit(1)

    if sum(args.percentages) > 1.0:
        print("Error: Percentages sum should not exceed 1.0")
        exit(1)

    split_dataset(args.dataset_path, args.output_path, args.percentages)
