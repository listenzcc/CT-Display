# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# %%
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import plotly.express as px
import plotly.graph_objects as go

from skimage import measure
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage.filters import maximum_filter

from config import CONFIG

from data_manager import list_subject_folders, dcm_files, get_image

# %%
# Prepare dynamic variables.
subject_folders = list_subject_folders()

# Maintain the dynamic information of the .dcm images
dynamic_dict = dict(
    raw_data_folder=CONFIG.get('CT_raw_data_folder'),
    subject_folders=subject_folders,
    subject_folder=subject_folders[1],
    subject_num_dcm_files=len(dcm_files(subject_folders[1])),
    subject_current_slice_idx=int(len(dcm_files(subject_folders[1])) / 2),
)


def dynamic_dict_report():
    '''
    Maintain the report of the dynamic_dict
    '''
    lines = ['{}: {}'.format(e, dynamic_dict[e])
             for e in dynamic_dict]

    print(pd.DataFrame(dynamic_dict))

    return lines


# Large dynamic variables, like 3D image and figs of slices
large_dynamic_dict = dict(
    img=get_image(dynamic_dict['subject_folder']),
    figs_slice='Very large figs of plotly',
    fig_contour='3D image of contour surface'
)


def redraw_images_to_figs():
    '''
    Update the [figs] of the large_dynamic_dict,
    use it for every time its 'img' changes.

    The image is properly calculated.
    '''

    range_color = (-1000, 2000)

    # Read the latest image and strip its negative values.
    img = large_dynamic_dict['img']
    img[img < 0] = 0

    # Remove the skull for calculation,
    # using the maximum_filter method.
    kernel = np.ones((5, 5, 5))
    img_contour = img.copy()
    mask = maximum_filter(img.copy(), footprint=np.ones((5, 5, 5)))
    img_contour[mask > 200] = 0

    # Remove the **small** nodes for better solution.
    mask = img_contour > 50
    mask = binary_erosion(mask, kernel)
    mask = binary_dilation(mask, kernel)
    img_contour[mask < 1] = 0

    # The figs is a list of slice views
    figs = []
    for j in tqdm(range(dynamic_dict['subject_num_dcm_files']), 'Redraw images'):
        # Two-layers are generated.

        # The background is the gray-scaled brain slice view.
        fig = px.imshow(img[j],
                        range_color=range_color,
                        color_continuous_scale='gray')

        # The upper layer is the contour of values between start=50 and end=100,
        # it is designed to be the detected object
        fig.add_trace(go.Contour(z=img_contour[j],
                                 showscale=False,
                                 hoverinfo='skip',
                                 line_width=2,
                                 contours=dict(
                                     start=50,
                                     end=100,
                                     size=25,
                                     coloring='lines',
                                     showlabels=True,
                                     labelfont=dict(size=12, color='white'))))

        fig.update_layout({'title': 'Slice: {}'.format(j),
                           'dragmode': 'drawclosedpath',
                           'newshape.line.color': 'cyan'})
        figs.append(fig)
        pass

    large_dynamic_dict['figs_slice'] = figs

    # The fig_contour is the 3D view of the contour surface
    data = []

    # Skull
    color = 'grey'
    verts, faces, normals, values = measure.marching_cubes(img,
                                                           500,
                                                           step_size=3)
    x, y, z = verts.T
    i, j, k = faces.T
    print(color, [e.shape for e in [x, y, z, i, j, k]])
    data.append(
        go.Mesh3d(x=x, y=y, z=z, color=color, opacity=0.2, i=i, j=j, k=k)
    )

    # Target
    if np.count_nonzero(img_contour > 50):
        color = 'purple'
        verts, faces, normals, values = measure.marching_cubes(img_contour,
                                                               50,
                                                               step_size=3)
        x, y, z = verts.T
        i, j, k = faces.T
        print(color, [e.shape for e in [x, y, z, i, j, k]])
        data.append(
            go.Mesh3d(x=x, y=y, z=z, color=color, opacity=0.3, i=i, j=j, k=k)
        )

    layout = dict(scene={'aspectmode': 'data'},
                  title=dynamic_dict['subject_folder'])
    fig = go.Figure(data, layout=layout)

    large_dynamic_dict['fig_contour'] = fig

    return 0


redraw_images_to_figs()

# %%
external_stylesheets = [
    "assets/debug-style.css",
    "assets/basic-style.css"
]
app = dash.Dash(CONFIG['app_name'], external_stylesheets=external_stylesheets)

# %% ----------------------------------------------------------------------------
control_panel_1 = html.Div(
    className='allow-debug',
    children=[
        html.Div(
            id='display-1',
            className='allow-debug',
            children='-----------------------'
        ),
        html.Div(
            className='allow-debug',
            style={'display': 'flex', 'flex-direction': 'row'},
            children=[
                html.Div(
                    className='allow-debug',
                    style={'padding': 10, 'flex': 1},
                    children=[
                        html.Label('Dropdown'),
                        dcc.Dropdown(
                            id='CT-raw-data-folder-selector',
                            clearable=False,
                            options=[
                                {'label': e, 'value': e}
                                for e in dynamic_dict['subject_folders']
                            ],
                            value=dynamic_dict['subject_folder']
                        ),

                        html.Br(),
                        html.Label('Multi-Select Dropdown'),
                        dcc.Dropdown(
                            options=[
                                {'label': 'Target', 'value': 'Tar'},
                                {'label': 'Background', 'value': 'Bac'},
                                {'label': 'Others', 'value': 'Oth'}
                            ],
                            value=['Tar', 'Bac'],
                            multi=True
                        ),

                        html.Br(),
                        html.Label('Radio Items'),
                        dcc.RadioItems(
                            options=[
                                {'label': 'Target', 'value': 'Tar'},
                                {'label': 'Background', 'value': 'Bac'},
                                {'label': 'Others', 'value': 'Oth'}
                            ],
                            value='Tar'
                        ),
                    ]),

                html.Div(
                    className='allow-debug',
                    style={'padding': 10, 'flex': 1},
                    children=[
                        html.Label('Checkboxes'),
                        dcc.Checklist(
                            options=[
                                {'label': 'Target', 'value': 'Tar'},
                                {'label': 'Background', 'value': 'Bac'},
                                {'label': 'Others', 'value': 'Oth'}
                            ],
                            value=['Tar', 'Bac'],
                        ),

                        html.Br(),
                        html.Label('Text Input'),
                        dcc.Input(value='Text', type='text'),

                        html.Br(),
                        html.Label('Slider'),
                        dcc.Slider(
                            id='slider-1',
                            min=0,
                            max=dynamic_dict['subject_num_dcm_files'],
                            marks={i: 'Slice {}'.format(i) if i == 0 else str(i)
                                   for i in range(0, dynamic_dict['subject_num_dcm_files'])},
                            value=dynamic_dict['subject_current_slice_idx'],
                            updatemode='drag'
                        )
                    ]
                )
            ]
        )
    ]
)


# %% ----------------------------------------------------------------------------
view_panel_1 = html.Div(
    id='view-panel-1',
    className='allow-debug',
    style={'display': 'flex', 'flex-direction': 'row'},
    children=[
        dcc.Graph(
            id='graph-2',
            figure=large_dynamic_dict['fig_contour']
        ),
        dcc.Graph(
            id='graph-1',
            figure=large_dynamic_dict['figs_slice'][dynamic_dict['subject_current_slice_idx']],
            config={
                "modeBarButtonsToAdd": [
                    "drawrect",
                    "drawopenpath",
                    "eraseshape",
                ]
            }
        )
    ]
)

# %% ----------------------------------------------------------------------------
message_panel_1 = html.Div(
    id='message-panel-1',
    className='allow-debug',
    children=[
        html.Div(
            id='report-area-1',
            className='allow-debug',
            children='Report Area 1'
        ),
        html.Div(
            id='report-area-2',
            className='allow-debug',
            children='Report Area 2'
        )
    ]
)

# %% ----------------------------------------------------------------------------
app.layout = html.Div(
    id='main-window',
    className='allow-debug',
    children=[
        html.Div(
            id='title-div',
            className='allow-debug',
            children=[
                html.H1(children='No Title')
            ]
        ),
        html.Div(
            id='control-div',
            className='allow-debug',
            children=[
                control_panel_1
            ]
        ),
        html.Div(
            id='view-div',
            className='allow-debug',
            children=[
                view_panel_1,
            ]
        ),
        html.Div(
            id='message-div',
            className='allow-debug',
            children=[message_panel_1]
        ),
    ]
)


# %% ----------------------------------------------------------------------------
@ app.callback(
    [
        Output('report-area-1', 'children'),
        Output('slider-1', 'marks'),
        Output('slider-1', 'min'),
        Output('slider-1', 'max'),
        Output('slider-1', 'value'),
        Output('graph-1', 'figure'),
        Output('graph-2', 'figure'),
    ],
    [
        Input('CT-raw-data-folder-selector', 'value'),
        Input('slider-1', 'value')
    ],
)
def callback_control_panel_1_1(subject_folder, slice_idx):
    print('-------------------------------------------------------------------')
    # fig = large_dynamic_dict['fig']
    # Which Input is inputted
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    print(cbcontext)

    # Update the dcm files
    num_dcm_files = len(dcm_files(subject_folder))
    dynamic_dict['subject_folder'] = subject_folder
    dynamic_dict['subject_num_dcm_files'] = num_dcm_files
    dynamic_dict['subject_current_slice_idx'] = slice_idx

    # Update the slider-1
    n = 1
    if num_dcm_files > 20:
        n = int(num_dcm_files / 10)
    marks = {i: 'Slice {}'.format(i) if i == 0 else str(i)
             for i in range(0, num_dcm_files, n)}
    min = 0
    max = num_dcm_files - 1

    # Auto change the slice_idx if new sub folder is selected
    if cbcontext == 'CT-raw-data-folder-selector.value':
        slice_idx = int(max/2)
        dynamic_dict['subject_current_slice_idx'] = slice_idx
        large_dynamic_dict['img'] = get_image(
            dynamic_dict['subject_folder'])
        redraw_images_to_figs()

    fig1 = large_dynamic_dict['figs_slice'][dynamic_dict['subject_current_slice_idx']]
    fig2 = large_dynamic_dict['fig_contour']

    report = ' | '.join(dynamic_dict_report())
    return report, marks, min, max, slice_idx, fig1, fig2


# %%
if __name__ == '__main__':
    app.run_server(debug=False)

# %%
