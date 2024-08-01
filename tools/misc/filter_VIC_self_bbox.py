import os
from tqdm import tqdm

label_path = "data/V2X-Seq-SPD-KITTI-CO/training/label_2"

# file = "010813.txt"
for file in tqdm(os.listdir(label_path)):
    with open(os.path.join(label_path, file), "r") as f:
        error_line = []
        lines = f.readlines()
        for i,line in enumerate(lines):
            line = line.strip().split(" ")
            x,y,z = float(line[11]), float(line[12]), float(line[13])
            if abs(x) < 2 and abs(y) < 2 and abs(z) < 2:
                error_line.append(i)
                # print(f"Error line {i}: {line}")    
        if len(error_line) > 1: 
            print(f"{file} has more than 1 error line: {error_line}")
    if len(error_line) > 0:        
        with open(os.path.join(label_path, file), "w") as f:
            for i,line in enumerate(lines):
                if i not in error_line:
                    f.write(line)
