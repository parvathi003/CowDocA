from vision.dataset import get_dataloaders

train_loader, valid_loader, test_loader, classes = get_dataloaders()

print("="*50)

print("Classes")

print(classes)

print()

print("Training Images :", len(train_loader.dataset))

print("Validation Images :", len(valid_loader.dataset))

print("Testing Images :", len(test_loader.dataset))

print("="*50)