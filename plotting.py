import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_route_paths(paths, coords, title: str, figsize=(10, 5)):
    figure, axis = plt.subplots(figsize=figsize)
    for label, path in paths.items():
        longitudes = [coords[airport]["longitude"] for airport in path]
        latitudes = [coords[airport]["latitude"] for airport in path]
        axis.plot(longitudes, latitudes, marker="o", linewidth=1.5, markersize=3, label=label)
    axis.set(title=title, xlabel="Longitude", ylabel="Latitude")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    return figure, axis
