import os


path = r'./data/plot_rt/plots'
all_files = os.listdir(path)

with open('./data/plot_rt/test_real.idl','w') as fh:
    fh.write('\n'.join(r'"plots/'+str(name)+r'";' for name in all_files))

fh.close()
