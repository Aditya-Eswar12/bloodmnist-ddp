"""Pack what the Colab notebook needs into colab_bundle.zip (code + data, about 160 MB)."""
import glob
import zipfile

with zipfile.ZipFile("colab_bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for path in glob.glob("src/*.py") + ["data/bloodmnist_64.npz", "data/compartments_64_p8.npz"]:
        z.write(path)
    for path in glob.glob("results/mae_*.json"):     # lets load_encoder read encoder sizes
        z.write(path)
print("wrote colab_bundle.zip")
