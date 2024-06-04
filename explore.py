import base64
import io
import json

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html, no_update
from PIL import Image

# data
output = np.load("embeddings/output_tsim_L_selfcauchy_ph.npy")
labels = np.load("embeddings/labels_tsim_L_selfcauchy_ph.npy")
fnames = np.load("embeddings/fnames_tsim_L_selfcauchy_ph.npy")
# from mimer to local format
fnames = np.array(
    ["planktons_dataset/data/" + "/".join(f.split("/")[-3:]) for f in fnames]
)

with open("label2id.json", "r") as f:
    label2id = json.load(f)
    id2label = {v: k for k, v in label2id.items()}

fig = go.Figure()
for i in range(103):
    if id2label[i] == "Bacillaria":
        fig.add_trace(
            go.Scattergl(
                x=output[labels == i, 0],
                y=output[labels == i, 1],
                mode="markers",
                marker=dict(
                    size=8, opacity=0.8, color=i, colorscale="portland", line_width=2
                ),
                name=id2label[i],
                # set id as embedding index
                ids=[str(idx) for idx in np.where(labels == i)[0]],
            )
        )
    else:
        fig.add_trace(
            go.Scattergl(
                x=output[labels == i, 0],
                y=output[labels == i, 1],
                mode="markers",
                marker=dict(size=3, opacity=0.4, color=i, colorscale="portland"),
                name=id2label[i],
                # set id as embedding index
                ids=[str(idx) for idx in np.where(labels == i)[0]],
            )
        )

fig.update_traces(hoverinfo="none", hovertemplate=None)
fig.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    autosize=False,
    height=600,
    width=1500,
)

app = Dash(__name__)
app.layout = html.Div(
    className="container",
    children=[
        dcc.Graph(
            id="graph",
            figure=fig,
            clear_on_unhover=True,
        ),
        dcc.Tooltip(id="tooltip"),
    ],
)


@callback(
    Output("tooltip", "show"),
    Output("tooltip", "bbox"),
    Output("tooltip", "children"),
    Input("graph", "hoverData"),
)
def display_hover_data(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    im_path = fnames[int(hover_data["id"])]
    im = Image.open(im_path)

    # base64 image
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    im_url = "data:image/png;base64,{}".format(encoded_image)

    children = [
        html.Div(
            [
                html.Img(src=im_url, style={"width": "400px"}),
                html.P(
                    f"{labels[int(hover_data['id'])]} - {id2label[labels[int(hover_data['id'])]]}"
                ),
            ]
        )
    ]
    return True, bbox, children


if __name__ == "__main__":
    app.run(debug=True)
