import os
import glob
import numpy as np
import argparse
import subprocess
import re
import time

def read_xyz(file_path):
    """
    从XYZ文件读取 (num_atoms, atoms, coords, smiles).
    假设XYZ格式规范：
      第一行是原子数，
      第二行为注释，其后 num_atoms 行为 原子符号及 x, y, z 坐标，
      最后一行为 SMILES 结构式。
    """
    with open(file_path, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]  # 去掉空行
    print(lines)
    num_atoms = int(lines[0])
    atoms = []
    coords = []
    for i in range(num_atoms):
        parts = lines[i + 2].split()
        atom = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append(atom)
        coords.append((x, y, z))
    # 最后一行为 SMILES 结构式
    smiles = lines[-1]
    return num_atoms, atoms, np.array(coords), smiles

def create_mopac_input(file_path, atoms, coords, output_directory):
    """
    为MOPAC生成输入文件。
    """
    base = os.path.splitext(os.path.basename(file_path))[0]
    input_file = os.path.abspath(os.path.join(output_directory, base + '.mop'))

    mol_str = "PM7 \n\n"
    mol_str += f"{base} - single point calculation\n"

    for atom, (x, y, z) in zip(atoms, coords):
        mol_str += f"{atom}   {x:.6f}   {y:.6f}   {z:.6f}\n"

    with open(input_file, 'w') as f:
        f.write(mol_str)

    return input_file

def run_mopac(input_file, output_directory):
    """
    运行MOPAC计算，并获取输出文件。
    """
    base = os.path.splitext(os.path.basename(input_file))[0]
    output_file = os.path.join(output_directory, base + '.out')

    subprocess.run(["mopac", input_file], cwd=output_directory)

    return output_file

def extract_results(output_file, atoms, coords, output_directory, smiles):
    """
    从MOPAC的输出文件中提取结果（能量、电荷等），
    并将优化后的结果写入XYZ文件，同时在最后一行写入原始XYZ文件中提取的 SMILES 结构式。
    """
    with open(output_file, 'r') as f:
        lines = f.readlines()

    scf_energy = None
    charges = []

    for i, line in enumerate(lines):
        if "FINAL HEAT OF FORMATION" in line:
            scf_energy = float(re.search(r'(-?\d+\.\d+)', line).group(1))
        if "NET ATOMIC CHARGES AND DIPOLE CONTRIBUTIONS" in line:
            start_index = i + 3
            for charge_line in lines[start_index:]:
                if not charge_line.strip() or "DIPOLE" in charge_line:
                    break
                parts = charge_line.split()
                charges.append(float(parts[2]))

    base = os.path.splitext(os.path.basename(output_file))[0]
    out_file = os.path.join(output_directory, base + ".xyz")

    with open(out_file, 'w') as outf:
        outf.write(f"{len(atoms)}\n")
        outf.write(f"{scf_energy:.8f}\n")
        for (atom, (x, y, z)), charge in zip(zip(atoms, coords), charges):
            outf.write(f"{atom:2s}  {x:12.6f}  {y:12.6f}  {z:12.6f}  {charge:12.6f}\n")
        # 在最后一行写入 SMILES 结构式
        outf.write(smiles + "\n")

    return scf_energy, charges

def main(xyz_directory, output_directory):
    start_time = time.time()

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    xyz_list = glob.glob(os.path.join(xyz_directory, '*.xyz'))
    print(f"在目录 {xyz_directory} 下找到 {len(xyz_list)} 个XYZ文件。")

    success_count = 0
    for fpath in xyz_list:
        num_atoms, atoms, coords, smiles = read_xyz(fpath)

        # 为MOPAC创建输入文件
        input_file = create_mopac_input(fpath, atoms, coords, output_directory)

        # 运行MOPAC计算
        output_file = run_mopac(input_file, output_directory)

        # 提取计算结果，并在优化后的XYZ文件中添加SMILES结构式
        scf_energy, charges = extract_results(output_file, atoms, coords, output_directory, smiles)

        print(f"[OK ] {fpath} - Total energy: {scf_energy:.8f}")
        success_count += 1

    print(f"处理完成：成功 {success_count}，失败 {len(xyz_list) - success_count}。")
    print(f"计算时间: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量对XYZ分子计算能量和电荷（使用MOPAC），并将SMILES结构式写入优化后的XYZ文件。")
    parser.add_argument('--xyz_dir', type=str, required=True, help="XYZ文件所在目录")
    parser.add_argument('--output_dir', type=str, required=True, help="输出目录")
    args = parser.parse_args()

    main(args.xyz_dir, args.output_dir)
