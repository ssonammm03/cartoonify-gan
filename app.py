import argparse, os
import torch, torchvision
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import Generator

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_args():
    p = argparse.ArgumentParser(description="Cartoonify one image")
    p.add_argument("--weights", type=str, default="C:\Users\user\Desktop\cartoonify-gan\runs\gen_e005.pt",
                   help="Path to generator weights (.pt)")
    p.add_argument("--input", type=str, default="C:/Users/user/Desktop/cartoonify-gan/unseen_data/outputs/images1.jpg",
                   help="Path to input image")
    p.add_argument("--img_size", type=int, default=128,
                   help="Resize to (img_size, img_size)")
    p.add_argument("--save", type=str, default="",
                   help="Optional path to save cartoonified image (e.g., outputs/result.png)")
    return p.parse_args()

def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)

def main():
    args = get_args()
    assert os.path.exists(args.weights), f"Missing weights: {args.weights}"
    assert os.path.exists(args.input), f"Missing input image: {args.input}"

    # 1️⃣ Load model
    gen = Generator().to(DEVICE)
    gen.load_state_dict(torch.load(args.weights, map_location=DEVICE))
    gen.eval()
    print(f"✅ Loaded weights: {args.weights}")

    # 2️⃣ Preprocess input
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    img = Image.open(args.input).convert("RGB")
    x = tf(img).unsqueeze(0).to(DEVICE)

    # 3️⃣ Generate cartoon
    with torch.no_grad():
        y = gen(x).cpu().squeeze(0)
    cartoon = denorm(y).permute(1, 2, 0).numpy()

    # 4️⃣ Optional save
    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torchvision.utils.save_image(denorm(y), args.save)
        print(f"💾 Saved: {args.save}")

    # 5️⃣ Display
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1); plt.imshow(img); plt.title("Original"); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(cartoon); plt.title("Cartoonified"); plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
