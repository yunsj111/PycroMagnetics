import seaborn as sns
import matplotlib.pyplot as plt

def show_mask(magnet, figsize=(5,5), fontsize=100):
    total_nz = magnet.mask.shape[0]
    d1 = total_nz // 5
    d2 = total_nz % 5
    if d2!=0:
        d1 += 1
    plt.figure(figsize=(figsize[0]*5, figsize[1]*d1),)
    for i in range(total_nz):
        plt.subplot(d1,5,i+1)
        sns.heatmap(magnet.mask[i].tolist(), cmap='gray', vmin=-1,vmax=1, cbar=False, yticklabels=False, xticklabels=False).invert_yaxis()
        plt.title('nz : {}/{}'.format(i,total_nz), fontdict={'size': fontsize})
    plt.show()
    
def show_MagProperties(mag_property, figsize=(5,5), fontsize=100):
    total_nz = mag_property.shape[0]
    d1 = total_nz // 5
    d2 = total_nz % 5
    if d2!=0:
        d1 += 1
    vmax = mag_property.max()
    plt.figure(figsize=(figsize[0]*5, figsize[1]*d1),)
    for i in range(total_nz):
        plt.subplot(d1,5,i+1)
        sns.heatmap(mag_property[i].tolist(), cmap='RdBu', vmin=-vmax, vmax=vmax, cbar=True, yticklabels=False, xticklabels=False).invert_yaxis()
        plt.title('nz : {}/{}'.format(i,total_nz), fontdict={'size': fontsize})
    plt.show()