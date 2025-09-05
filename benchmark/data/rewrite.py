import os
folder = '/share/home/alpha/block_predict/block_predict/benchmark/data/qm7_xyz'
for xyz in os.listdir(folder):
    abpath = os.path.join(folder, xyz)
    with open(abpath, 'r') as f:
        lines = f.readlines()
        prop_line = lines[1].strip().split(' ')
        prop =  prop_line[2]
        coordinates = lines[2:]
    with open(abpath, 'w') as f:
        f.write(f'{prop}\n')
        for line in coordinates:
            f.write(line)
            