import matplotlib.pyplot as plt
from collections import OrderedDict
import seaborn as sns
import numpy as np

"""
选择其中之一的颜色。各个颜色的渐变色图参见
https://matplotlib.org/stable/tutorials/colors/colormaps.html

渐变色图的展示参见
https://xercis.blog.csdn.net/article/details/88535217

color_name = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
"""

color_name = 'BuGn'

# 渐变色可视化出来看一下
plt.figure()
sns.palplot(sns.color_palette(color_name, 8))
plt.show()

# 给定一个[0,1]的浮点数，即可获得一个渐变色中的RGBA值(RGB值需要再乘255)
linear_colors = plt.get_cmap(color_name)
float2color = linear_colors(0.3)
print(float2color)



# 把各种渐变色条都可视化出来
cmaps = OrderedDict()
'''将颜色替换此处'''
cmaps['Sequential'] = ['BuGn', 'YlGn', 'Greens', 'Reds', 'Greens', 'Reds', 'Greens', 'Reds']
'''将颜色替换此处'''

nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps.items())
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

def plot_color_gradients(cmap_category, cmap_list, nrows):
    fig, axes = plt.subplots(nrows=nrows)
    fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
    axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

    for ax, name in zip(axes, cmap_list):
        ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
        pos = list(ax.get_position().bounds)
        x_text = pos[0] - 0.01
        y_text = pos[1] + pos[3]/2.
        fig.text(x_text, y_text, name, va='center', ha='right', fontsize=10)

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axes:
        ax.set_axis_off()

for cmap_category, cmap_list in cmaps.items():
    plot_color_gradients(cmap_category, cmap_list, nrows)

plt.show()

