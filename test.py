from data import mnist
import matplotlib.pyplot as plt

train, _ = mnist()

images, labels = next(iter(train))

print(labels[0])
plt.imshow(images[0].squeeze(), cmap='Greys_r')
plt.show()