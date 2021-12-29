# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

# %%
import dash
from dash import dcc
from dash import html
from dash import dash_table
from dash.dependencies import Input, Output

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import plotly.express as px
import plotly.graph_objects as go

import SimpleITK as sitk
import radiomics
from radiomics import featureextractor, getTestCase

from skimage import measure
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage.filters import maximum_filter

from config import CONFIG, logger

from data_manager import DCM_Manager


# %%
logger.info('Dash App is initializing.')

# Prepare dynamic variables.
dcm = DCM_Manager()
subject_folders = dcm.list_subject_folders()

# Maintain the dynamic information of the .dcm images
dynamic_dict = dict(
    raw_data_folder=CONFIG.get('CT_raw_data_folder'),
    subject_folders=subject_folders,
    subject_folder=subject_folders[1],
    subject_num_dcm_files=len(dcm.dcm_files(subject_folders[1])),
    subject_current_slice_idx=int(len(dcm.dcm_files(subject_folders[1])) / 2),
)


def dynamic_dict_report():
    '''
    Maintain the report of the dynamic_dict
    '''
    lines = ['{}: {}'.format(e, dynamic_dict[e])
             for e in dynamic_dict]

    logger.debug('Current dynamic_dict is {}'.format(
        pd.DataFrame(dynamic_dict)))

    return lines


# Large dynamic variables, like 3D image and figs of slices
large_dynamic_dict = dict(
    img=dcm.get_image(dynamic_dict['subject_folder']),
    img_contour='The contour version of the img, Updated by redraw_imgs_to_figs',
    figs_slice='Very large figs of plotly, Updated by redraw_imgs_to_figs',
    fig_contour='3D image of contour surface, Updated by redraw_imgs_to_figs',
    table_obj='The table object, Updated by redraw_imgs_to_figs'
)


def compute_features():
    ''' Compute the features using the current data '''
    subject_folder = dynamic_dict['subject_folder']
    img_contour = large_dynamic_dict['img_contour'].copy()
    img_contour[img_contour > 0] = 1

    # Return empty if can NOT find valid regions
    if np.count_nonzero(img_contour) == 0:
        df = pd.DataFrame([{'subject_folder': subject_folder}])
        logger.debug('No valid regions found.')
        return df

    logger.debug('The img_contour contains {} nonzero pixels'.format(
        np.count_nonzero(img_contour)))

    # Get image as sitk format
    reader = sitk.ImageSeriesReader()
    names = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(dcm.current_data_folder)
    reader.SetFileNames(names)
    image = reader.Execute()
    logger.debug(
        'Got image (sitk) with the shape of {}.'.format(image.GetSize()))

    # Compute features
    img_mask = sitk.GetImageFromArray(img_contour)
    rfe = featureextractor.RadiomicsFeatureExtractor()
    mask = radiomics.imageoperations.getMask(img_mask)
    rfe.loadImage(image, mask)
    logger.debug('The featureextractor:"{}" loaded image and mask'.format(rfe))

    features = rfe.computeFeatures(image, mask, 'original')
    lst = []
    for name in tqdm(features, 'Collecting Features'):
        lst.append((name, features[name]))

    df = pd.DataFrame(lst, columns=['name', 'value'])
    df['subject_folder'] = subject_folder

    logger.debug('Computed features for {} entries.'.format(len(df)))
    return df


def compute_contour(img):
    ''' Process the raw img to compute img_contour '''
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

    return img_contour


def redraw_images_to_figs():
    '''
    Update the [figs] of the large_dynamic_dict,
    use it for every time its 'img' changes.

    The image is properly calculated.
    '''

    # --------------------------------------------------------------------------------
    # Read the latest image and strip its negative values.
    img = large_dynamic_dict['img']
    img[img < 0] = 0

    img_contour = compute_contour(img)
    large_dynamic_dict['img_contour'] = img_contour
    logger.debug('The large_dynamic_dict.img_contour is updated.')

    # --------------------------------------------------------------------------------
    # The figs is a list of slice views
    figs = []
    range_color = (-1000, 2000)
    for j in tqdm(range(dynamic_dict['subject_num_dcm_files']), 'Redraw images'):
        # Two-layers will be generated.
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
    logger.debug('The large_dynamic_dict.figs_slice is updated.')

    # --------------------------------------------------------------------------------
    # The fig_contour is the 3D view of the contour surface
    data = []

    # Skull
    color = 'grey'
    verts, faces, normals, values = measure.marching_cubes(img,
                                                           500,
                                                           step_size=3)
    x, y, z = verts.T
    i, j, k = faces.T

    logger.debug('Using color: "{}" rendering vertex with shape {}'.format(
        color, [e.shape for e in [x, y, z, i, j, k]]))

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

        logger.debug('Using color: "{}" rendering vertex with shape {}'.format(
            color, [e.shape for e in [x, y, z, i, j, k]]))

        data.append(
            go.Mesh3d(x=x, y=y, z=z, color=color, opacity=0.3, i=i, j=j, k=k)
        )

    layout = dict(scene={'aspectmode': 'data'},
                  title=dynamic_dict['subject_folder'])
    fig = go.Figure(data, layout=layout)

    large_dynamic_dict['fig_contour'] = fig
    logger.debug('The large_dynamic_dict.fig_contour is updated.')

    # --------------------------------------------------------------------------------
    df = compute_features()
    columns = [{"name": i, "id": i} for i in df.columns]
    data = df.to_dict('records')

    table_obj = dash_table.DataTable(
        columns=columns,
        data=data
    )

    large_dynamic_dict['table_obj'] = table_obj
    logger.debug('The large_dynamic_dict.table_obj is updated')

    return 0


redraw_images_to_figs()


logger.info('Dash App estalished the requires.')

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

                        # html.Br(),
                        # html.Label('Text Input'),
                        # dcc.Input(value='Text', type='text'),

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
                        ),

                        html.Br(),
                        html.Label('Feature'),
                        html.Button('Compute', id='button-1', n_clicks=0),
                    ]
                )
            ]
        )
    ]
)


# %% ----------------------------------------------------------------------------
view_panel_1 = dcc.Loading(html.Div(
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
), type='circle')

# %% ----------------------------------------------------------------------------

message_panel_1 = dcc.Loading(html.Div(
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
            children='Very Good Table',
        )
    ]
), type='circle')

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
                # message_panel_1,
            ]
        ),
        html.Div(
            id='message-div',
            className='allow-debug',
            children=[
                message_panel_1,
            ]
        ),
    ]
)

logger.info('Dash App estalished the html layout.')

# %% ----------------------------------------------------------------------------


# @ app.callback(
#     Output('report-area-2', 'children'),
#     Input('button-1', 'n_clicks'),
# )
# def callback_button_1_1(n_clicks):
#     cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
#     logger.debug(
#         'The callback_button_1_1 receives the event: {}'.format(cbcontext))

#     df = compute_features()
#     columns = [{"name": i, "id": i} for i in df.columns]
#     data = df.to_dict('records')
#     print(columns, data)

#     table_obj = dash_table.DataTable(
#         id='table-1',
#         columns=columns,
#         data=data
#     )

#     return table_obj,


@ app.callback(
    [
        Output('report-area-1', 'children'),
        Output('slider-1', 'marks'),
        Output('slider-1', 'min'),
        Output('slider-1', 'max'),
        Output('slider-1', 'value'),
        Output('graph-1', 'figure'),
        Output('graph-2', 'figure'),
        Output('report-area-2', 'children'),
    ],
    [
        Input('CT-raw-data-folder-selector', 'value'),
        Input('slider-1', 'value')
    ],
)
def callback_control_panel_1_1(subject_folder, slice_idx):
    # --------------------------------------------------------------------------------
    # Which Input is inputted
    cbcontext = [p["prop_id"] for p in dash.callback_context.triggered][0]
    logger.debug(
        'The callback_control_panel_1_1 receives the event: {}'.format(cbcontext))

    # --------------------------------------------------------------------------------
    # Update the dcm files
    num_dcm_files = len(dcm.dcm_files(subject_folder))
    dynamic_dict['subject_folder'] = subject_folder
    dynamic_dict['subject_num_dcm_files'] = num_dcm_files
    dynamic_dict['subject_current_slice_idx'] = slice_idx
    logger.debug('The dynamic_dict is updated since it is quick.')

    # --------------------------------------------------------------------------------
    # Update the slider-1
    n = 1
    if num_dcm_files > 20:
        n = int(num_dcm_files / 10)
    marks = {i: 'Slice {}'.format(i) if i == 0 else str(i)
             for i in range(0, num_dcm_files, n)}
    min = 0
    max = num_dcm_files - 1
    logger.debug('The marks, min and max parameters for slider-1 is updated.')

    # --------------------------------------------------------------------------------
    # Redraw the figures if the new subject-folder is selected.
    # Auto change the slice_idx if new subject-folder is selected.
    if cbcontext == 'CT-raw-data-folder-selector.value':
        slice_idx = int(max/2)
        dynamic_dict['subject_current_slice_idx'] = slice_idx
        large_dynamic_dict['img'] = dcm.get_image(
            dynamic_dict['subject_folder'])
        redraw_images_to_figs()
        logger.debug(
            'Figures are redrawn since the new subject-folder:{} is selected'.format(subject_folder))
        logger.debug(
            'The slice_idx is automatically updated to {}.'.format(slice_idx))

    # --------------------------------------------------------------------------------
    # Make the figures and report for update the html components
    fig1 = large_dynamic_dict['figs_slice'][dynamic_dict['subject_current_slice_idx']]
    fig2 = large_dynamic_dict['fig_contour']
    report = ' | '.join(dynamic_dict_report())
    table_obj = large_dynamic_dict['table_obj']
    logger.debug('The new figures and report are generated and returned.')

    return report, marks, min, max, slice_idx, fig1, fig2, table_obj


logger.info('Dash App estalished the callbacks.')

logger.info('Dash App initialized.')

# %%
if __name__ == '__main__':
    logger.info('Dash App is starting its serve.')
    app.run_server(debug=True)

# %%
