from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# ==========================================
# Image Size
# ==========================================

IMAGE_SIZE = 224

# ==========================================
# Training Transform
# ==========================================

train_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(15),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    )

])

# ==========================================
# Validation/Test Transform
# ==========================================

test_transform = transforms.Compose([

    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[0.485, 0.456, 0.406],

        std=[0.229, 0.224, 0.225]

    )

])

# ==========================================
# Load Dataset
# ==========================================

def get_dataloaders():

    train_dataset = datasets.ImageFolder(

        "dataset/train",

        transform=train_transform

    )

    valid_dataset = datasets.ImageFolder(

        "dataset/valid",

        transform=test_transform

    )

    test_dataset = datasets.ImageFolder(

        "dataset/test",

        transform=test_transform

    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=32,

        shuffle=True

    )

    valid_loader = DataLoader(

        valid_dataset,

        batch_size=32,

        shuffle=False

    )

    test_loader = DataLoader(

        test_dataset,

        batch_size=32,

        shuffle=False

    )

    return (

        train_loader,

        valid_loader,

        test_loader,

        train_dataset.classes

    )