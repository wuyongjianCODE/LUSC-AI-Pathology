import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset_names", default="all", type=str) # "all" or names joined by comma
argparser.add_argument("--dataset_path", default="DATASET/odinw35", type=str)
args = argparser.parse_args()

root = "https://vlpdatasets.blob.core.windows.net/odinw/odinw/odinw_35"

all_datasets = ["AerialMaritimeDrone", "Aquarium","CottontailRabbits", "EgoHa", "NorthAmericaMushrooms",  "Packages", "PascalVOC", "Raccoon", "ShellfishOpenImages", "ThermalCheetah", "VehiclesOpenImages", "pist",  "poth"]

datasets_to_download = []
if args.dataset_names == "all":
    datasets_to_download = all_datasets
else:
    datasets_to_download = args.dataset_names.split(",")

for dataset in datasets_to_download:
    if dataset in all_datasets and not os.path.exists('DATASET/odinw35/{}'.format(dataset,dataset)):
        print("Downloading dataset: ", dataset)
        os.system('bypy download test/ODinW/{}.zip DATASET/odinw35/{}.zip'.format(dataset,dataset))
        # os.system("wget " + root + "/" + dataset + ".zip" + " -O " + args.dataset_path + "/" + dataset + ".zip")
        os.system("unzip " + args.dataset_path + "/" + dataset + ".zip -d " + args.dataset_path)
        os.system("rm " + args.dataset_path + "/" + dataset + ".zip")
    else:
        print("Dataset not found: ", dataset)
