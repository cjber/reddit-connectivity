import geopandas as gpd
import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("agg")

h3_poly = gpd.read_parquet("./data/out/h3_poly.parquet")
h3_pci = pd.read_parquet("./data/out/h3_pci.parquet").set_index("target_h3_05")
h3_pci = h3_pci.join(h3_poly)
h3_pci = gpd.GeoDataFrame(h3_pci, geometry="geometry")


def filter_gdf(name):
    name = name.lower()
    h3_filtered = h3_pci[h3_pci["name"].str.lower() == name]

    fig, ax = plt.subplots(1, figsize=(4, 8))
    ax.axis("off")

    h3_filtered.plot(ax=ax, column="PCI")
    return fig


with gr.Blocks() as demo:
    with gr.Row():
        name = gr.Text(value="Liverpool")
        btn = gr.Button(value="Update Filter")
        fig = gr.Plot(label=name)

    demo.load(filter_gdf, [name], fig)
    btn.click(filter_gdf, [name], fig)

demo.launch(height=4)
