import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

h3_poly = gpd.read_parquet("./data/out/h3_poly.parquet")
h3_pci = pd.read_parquet("./data/out/pci_h3.parquet").set_index("target_h3_05")
h3_pci = h3_pci.join(h3_poly)
h3_pci = gpd.GeoDataFrame(h3_pci, geometry="geometry")

padding = 0
# st.set_page_config(page_title="Cognitive Place Relations", layout="wide", page_icon="📍")

st.markdown(
    """
    <style>
    .small-font {
        font-size:12px;
        font-style: italic;
        color: #b1a7a6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
TABLE_PAGE_LEN = 10

st.title("Cognitive Place Connections")


with st.sidebar.form(key="my_form"):
    selectbox_state = st.text_input("Type a location!", value="Liverpool")
    st.markdown(
        '<p class="small-font">Results Limited to top 5 per State in overall US</p>',
        unsafe_allow_html=True,
    )
    pressed = st.form_submit_button("Process")

expander = st.sidebar.expander("What is this?")
expander.write(
    """
This app allows users to view migration between states from 2018-2019.
Overall US plots all states with substantial migration-based relationships with other states.
Any other option plots only migration from or to a given state. This map will be updated
to show migration between 2019 and 2020 once new census data comes out.

Incoming: Shows for a given state, the percent of their **total inbound migration from** another state.

Outgoing: Shows for a given state, the percent of their **total outbound migration to** another state.
"""
)


def filter_gdf(name):
    h3_filtered = h3_pci[h3_pci["top_h3_word"].str.lower() == name.lower()]

    if len(h3_filtered) > 0:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.axis("off")

        h3_filtered.plot(ax=ax, column="PCI")
        return fig
    else:
        st.write("Error! No locations with that name!")



name = "Liverpool"
st.title(f"{name}")

st.write(
    """
    Hope you like the map!
    """
)

if pressed:
    st.pyplot(filter_gdf(selectbox_state))
    plt.show()
