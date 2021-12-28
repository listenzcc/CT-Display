# %%
import SimpleITK as sitk
import pandas as pd
import numpy as np

from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage.filters import maximum_filter

from pathlib import Path
from tqdm.auto import tqdm

import radiomics
from radiomics import featureextractor, getTestCase

# %%
folders = ['s1-pre', 's1-post', 's2-pre', 's2-post']
lsts = []

# %%
for folder in tqdm(folders):
    data_dir = Path('../CT-data').joinpath(folder)
    print(data_dir)

    # -- %%
    reader = sitk.ImageSeriesReader()
    names = sitk.ImageSeriesReader().GetGDCMSeriesFileNames(str(data_dir))
    reader.SetFileNames(names)
    image = reader.Execute()
    image.GetSize()

    # -- %%

    img_list = [sitk.GetArrayFromImage(sitk.ReadImage(e))
                for e in tqdm(names, 'Reading .dcm files')]
    img = np.concatenate(img_list, axis=0)

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
    img_contour[img_contour > 0] = 1

    if np.count_nonzero(img_contour) == 0:
        lsts.append([])
        continue

    img_mask = sitk.GetImageFromArray(img_contour)

    # -- %%
    rfe = featureextractor.RadiomicsFeatureExtractor()
    # , label=np.unique(img_mask))
    mask = radiomics.imageoperations.getMask(img_mask)
    rfe.loadImage(image, mask)

    features = rfe.computeFeatures(image, mask, 'original')
    lst = []
    for name in features:
        lst.append((name, features[name]))
        print(lst[-1])

    lsts.append(lst)

# %%
df = pd.DataFrame(lsts[0], columns=['name', 'value'])
for j, folder in enumerate(folders):
    if lsts[j]:
        df[folder] = [e[1] for e in lsts[j]]
    else:
        df[folder] = ['--' for _ in range(len(df))]

df.drop('value', axis=1, inplace=True)
df.to_csv('feature.csv')
df

# %%
dir(image)

# %%
