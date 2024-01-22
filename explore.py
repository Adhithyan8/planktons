import base64
import io

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, callback, dcc, html, no_update
from datasets import load_dataset
from PIL import Image

# dataset
train_dataset = load_dataset(
    "planktons_dataset", "2013-14", split="train", trust_remote_code=True
)
# embeddings
embeddings = np.load("embeddings/tsne_resnet50.npy")
embeddings = embeddings[: len(train_dataset)]

# labels
names = train_dataset.features["label"].names
indices = np.load("embeddings/tags.npy").astype(int)
indices = indices[: len(train_dataset)]
labels = [names[idx] for idx in indices]

# set colors
colormap = [f"hsl({h},50%, 50%)" for h in np.linspace(0, 360, max(indices) + 1)]
colors = np.array([colormap[idx] for idx in indices])

fig = go.Figure()
for i in range(max(indices) + 1):
    fig.add_trace(
        go.Scattergl(
            x=embeddings[indices == i, 0],
            y=embeddings[indices == i, 1],
            mode="markers",
            marker=dict(size=2, opacity=0.8, color=colors[indices == i]),
            name=names[i],
            # set id as embedding index
            ids=[str(idx) for idx in np.where(indices == i)[0]],
        )
    )

fig.update_traces(hoverinfo="none", hovertemplate=None)
fig.update_layout(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    autosize=False,
    height=1400,
    width=2000,
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
    im_path = train_dataset[int(hover_data["id"])]["image"].filename
    im = Image.open(im_path)

    # base64 image
    buffer = io.BytesIO()
    im.save(buffer, format="PNG")
    encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    im_url = "data:image/png;base64,{}".format(encoded_image)

    children = [
        html.Div(
            [
                html.Img(src=im_url, style={"width": "200px"}),
                html.P(
                    f"{indices[int(hover_data['id'])]} - {labels[int(hover_data['id'])]}"
                ),
            ]
        )
    ]
    return True, bbox, children


if __name__ == "__main__":
    app.run(debug=True)
