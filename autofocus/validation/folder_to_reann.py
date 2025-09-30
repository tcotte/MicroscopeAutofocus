import os
import shutil

input_folder = r'D:\03 - IDEA\Micronoyaux\Autofocus\datasets_v2\X\slide_3'
output_folder = r'D:\03 - IDEA\Micronoyaux\Autofocus\datasets_to_reann\X\slide_3'

os.makedirs(output_folder)

if __name__ == '__main__':
    with open(r"C:\Users\tristan_cotte\Downloads\folder_to_reann.txt","r") as f:
        list_lines = f.readlines()

    for line in list_lines:
        x_pos, y_pos = os.path.basename(line).split('_')[:2]
        dirname = f'{x_pos}x_{y_pos}y'
        dir_ = os.path.join(input_folder, dirname)
        shutil.copytree(dir_, os.path.join(output_folder, dirname))