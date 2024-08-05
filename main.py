from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("img.png")
print(img)

trans_to_tensor = transforms.ToTensor()
img_tensor = trans_to_tensor(img)
# print(img_tensor)
writer.add_image("ToTensor", img_tensor)

# Normalize
trans_norm = transforms.Normalize([0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)
# print(img_norm)
writer.add_image("Normalize", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_to_tensor(img_resize)
writer.add_image("Resize", img_resize)
# print(img_resize)

# Compose
trans_compose = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((512, 512))
])
img_transpose = trans_compose(img)
writer.add_image("Compose", img_transpose)

# RandomCrop
trans_random = transforms.RandomCrop(15)
trans_compose_2 = transforms.Compose([trans_to_tensor, trans_random])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
