import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

def show_three_images(img_lq, img_gt, output):
    # resize to 256×256
    img_lq = TF.resize(img_lq, [256, 256])
    img_gt = TF.resize(img_gt, [256, 256])
    output = TF.resize(output, [256, 256])

    # remove batch dimension and convert to numpy
    imgs = [img_lq[0], img_gt[0], output[0]]
    imgs = [img.permute(1, 2, 0).detach().cpu().numpy() for img in imgs]

    # plot in a row
    titles = ['LQ', 'GT', 'Output']
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()