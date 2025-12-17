import numpy as np

a = np.array([[1, 2, 3],
              [4, 5, 6]])

print(a[1:4])      # row slice start at row 1 and stop at row 4. but 3rd, 4th row does not exist.
print(a[:, 1])     # all rows, column 1


import numpy as np

cube = np.random.rand(4, 4, 4)

print(cube[..., 0])


import numpy as np

arr = np.array([1, 2, 3])

print(arr[:, np.newaxis]) #np.newaxis convert 1D array into column vector.from shape(3,) to (3,1)


import numpy as np
from scipy.sparse import dia_matrix

d = np.array([[3, 0, 0, 0, 0], [0, 5, 0, 0, 0]])
offsets = np.array([0, -1])  
dia = dia_matrix((d, offsets), shape=(4, 5))

print(dia.toarray())


# ///////////////////////////////////////////////
from skimage.color import rgb2gray, rgba2rgb
from scipy.ndimage import gaussian_filter
import imageio.v3 as iio
import matplotlib.pyplot as plt

img = iio.imread('raccoon.png')
if img.shape[-1] == 4:
    img = rgba2rgb(img)

gray = rgb2gray(img).astype(float)
blur = gaussian_filter(gray, 5)
alpha = 30
sharp = gray + alpha * (gray - gaussian_filter(blur, 1))

plt.imshow(sharp, cmap='gray')
plt.axis('off')
plt.title("Sharpened Image")
plt.show()

#////////////////////////////////////////////

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, gaussian_filter, sobel

im = np.zeros((300, 300))
im[64:-64, 64:-64] = 1

im = rotate(im, 30, mode='constant')
im = gaussian_filter(im, sigma=7)

plt.imshow(im, cmap='gray')
plt.axis('off')
plt.title("Original Synthetic Image")
plt.show()

dx = sobel(im, axis=0, mode='constant')
dy = sobel(im, axis=1, mode='constant')
sobel_edges = np.hypot(dx, dy)

plt.imshow(sobel_edges, cmap='gray')
plt.axis('off')
plt.title("Sobel Edge Detection")
plt.show()


# In order to access multiple elements from a series, we use Slice operation. Slice operation is performed on Series with the use of the colon(:). To print elements from beginning to a range use [:Index], to print elements from end-use [:-Index], to print elements from specific Index till the end use [Index:], to print elements within a range, use [Start Index:End Index] and to print whole Series with the use of slicing operation, use [:]. Further, to print the whole Series in reverse order, use [::-1].



import json  
import pandas as pd  
from pandas import json_normalize  

with open('/content/raw_nyc_phil.json') as f:
    d = json.load(f)

nycphil = json_normalize(d['programs'])
nycphil.head(3)


# ///////////////////////////////////////////////////////

#Creating dataframe1
df1 = pd.DataFrame({
        'Name':['Jeevan', 'Raavan', 'Geeta', 'Bheem'], 
        'Age':[25, 24, 52, 40], 
        'Qualification':['Msc', 'MA', 'MCA', 'Phd']})
df1


#Creating dataframe2
df2 = pd.DataFrame({'Name':['Jeevan', 'Raavan', 'Geeta', 'Bheem'],
                    'Salary':[100000, 50000, 20000, 40000]})
df2

#Merging two dataframes
df = pd.merge(df1, df2)
df

# Apply function
def fun(value):
    if value > 70:
        return "Yes"
    else:
        return "No"

data_frame['Customer Satisfaction'] = data_frame['Spending Score (1-100)'].apply(fun)
data_frame.head(10)


const = data_frame['Age'].max()
data_frame['Age'] = data_frame['Age'].apply(lambda x: x/const)
data_frame.head()
# /////////////////////////////////////////////////////////////

# pandas correlation

#ADF stationary

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

# Creating dataset
np.random.seed(23685752)
N_points = 10000
n_bins = 20

# Creating distribution
x = np.random.randn(N_points)
y = 0.8 ** x + np.random.randn(N_points) + 25
legend = ['distribution']

# Creating figure and axes
fig, axs = plt.subplots(1, 1, figsize=(10, 7), tight_layout=True)

# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    axs.spines[s].set_visible(False)

# Remove x, y ticks
axs.xaxis.set_ticks_position('none')
axs.yaxis.set_ticks_position('none')

# Add padding between axes and labels
axs.xaxis.set_tick_params(pad=5)
axs.yaxis.set_tick_params(pad=10)

# Add x, y gridlines (updated syntax)
axs.grid(visible=True, color='grey', linestyle='-.', linewidth=0.5, alpha=0.6)

# Add text watermark
fig.text(0.9, 0.15, 'Jeeteshgavande30',
         fontsize=12,
         color='red',
         ha='right',
         va='bottom',
         alpha=0.7)

# Creating histogram
N, bins, patches = axs.hist(x, bins=n_bins)

# Setting color gradient
fracs = ((N ** (1 / 5)) / N.max())
norm = colors.Normalize(fracs.min(), fracs.max())

for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)

# Adding extra features
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend(legend)
plt.title('Customized Histogram with Watermark')

# Show plot
plt.show()

# ---------------------------------

# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

# Creating dataset
x = np.outer(np.linspace(-3, 3, 32), np.ones(32))
y = x.copy().T # transpose
z = (np.sin(x **2) + np.cos(y **2) )

# Creating figure
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')

# Creating color map
my_cmap = plt.get_cmap('hot')

# Creating plot
surf = ax.plot_surface(x, y, z,
                       cmap = my_cmap,
                       edgecolor ='none')

fig.colorbar(surf, ax = ax,
             shrink = 0.5, aspect = 5)

ax.set_title('Surface plot')

# show plot
plt.show()