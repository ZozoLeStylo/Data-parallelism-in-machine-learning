import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import time
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import csv
import os

def run(rank, size):
    # Configuration des paramètres
    model_name = "swin_b"
    batch_size = 64

    # Configurer le dispositif GPU
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device_id = rank % torch.cuda.device_count()

    # Prépare les transformations pour les données
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Chargement et division des données
    dataset = torchvision.datasets.Imagenette('/data/neo/user/chxu/', transform=transform_train)
    dataset_size = len(dataset)
    localdataset_size = dataset_size // size
    local_dataset = torch.utils.data.Subset(dataset, range(rank * localdataset_size, (rank + 1) * localdataset_size))
    dataloader = DataLoader(local_dataset, batch_size=batch_size // size, shuffle=True)

    # Charger le modèle choisi
    model = getattr(models, model_name)().to(device_id)

    # Modifier la couche finale pour adapter au dataset actuel
    if hasattr(model, "head"):
        model.head = nn.Linear(model.head.in_features, len(dataset.classes)).to(device_id)
    elif hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, len(dataset.classes)).to(device_id)
    else:
        raise AttributeError(f"The model {model_name} does not have a recognizable head or fc layer.")

    # Envelopper dans DistributedDataParallel
    ddp_model = DDP(model, device_ids=[device_id])

    # Configurer l'optimiseur et la fonction de perte
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # Entraînement
    try:
        st = time.time()
        train_images, train_labels = next(iter(dataloader))
        train_images = train_images.to(device_id)
        train_labels = train_labels.to(device_id)
        et_read = time.time()
        optimizer.zero_grad()
        outputs = ddp_model(train_images)
        loss_fn(outputs, train_labels).backward()
        et = time.time()
        optimizer.step()

        # Écriture dans un fichier CSV unique
        with open("results_gpu.csv", "a") as f:
            writer = csv.writer(f)
            writer.writerow([size, rank, batch_size, et_read - st, et - et_read])
    finally:
        # Assurez-vous que le groupe est détruit
        dist.destroy_process_group()


if __name__ == "__main__":
    dist.init_process_group("nccl", init_method="env://")
    size = dist.get_world_size()
    rank = dist.get_rank()
    run(rank, size)
