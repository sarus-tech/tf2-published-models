import matplotlib.pyplot as plt

from biggan import BigGAN

model = BigGAN.from_pretrained("biggan-deep-128")

label = "lion"
images = model.sample_with_labels(labels=[label], truncation=0.7)
img = (images[0].numpy() + 1.0) * 0.5
plt.imshow(img)
plt.axis("off")
plt.suptitle(label)
plt.savefig(f"{label}.png")
