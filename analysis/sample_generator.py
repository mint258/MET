#!/usr/bin/env python
# -*- coding: utf-8 -*-

from rdkit import Chem
from rdkit.Chem import AllChem
import os

def generate_xyz(mol, comment, smiles):
    """
    根据分子 mol 的第一构象生成 xyz 格式文本。
    第一行为原子数，第二行为注释（此处记录分子类别），最后一行为 SMILES 结构式。
    """
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    lines = []
    lines.append(str(num_atoms))
    lines.append(comment)
    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        symbol = atom.GetSymbol()
        # 保留 4 位小数
        lines.append(f"{symbol} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}")
    # 将 SMILES 结构式添加为最后一行
    lines.append(smiles)
    return "\n".join(lines)

def main():
    # 定义各官能团对应的 SMILES 列表
    # molecules = {
    #     "Benzene": [  # 苯（允许多个苯环或苯衍生的甲基连接）
    #         "c1ccccc1",                           # 1. 苯
    #         "Cc1ccccc1",                          # 2.
    #         "CCc1ccccc1",                         # 3.
    #         "CCCc1ccccc1",                        # 4.
    #         "CC(C)c1ccccc1",                       # 5.
    #         "c1ccccc1C(c2ccccc2)",                  # 6.
    #         "C(c1ccccc1)(c2ccccc2)c3ccccc3",         # 7. 三苯甲烷
    #         "c1ccccc1Cc2ccccc2",                    # 8.
    #         "c1ccc(-c2ccccc2)cc1",                  # 9. 二苯
    #         "c1ccccc1CCc2ccccc2",                   # 10.
    #         "CCc1ccccc1Cc2ccccc2",                  # 11.
    #         "Cc1ccccc1C(C)c2ccccc2",                # 12.
    #         "c1ccccc1C(C)c2ccccc2",                 # 13.
    #         "c1ccccc1C(C)(C)c2ccccc2",              # 14.
    #         "C(c1ccccc1)(c2ccccc2)C(C)c3ccccc3",     # 15.
    #         "c1ccc(cc1)C(c2ccccc2)",                # 16.
    #         "c1ccccc1CC(c2ccccc2)C",                # 17.
    #         "CC(c1ccccc1)C(c2ccccc2)c3ccccc3",       # 18.
    #         "c1ccc(-c2ccccc2)c(c1)C",               # 19.
    #         "c1ccc(cc1)C(c2ccccc2)C(c3ccccc3)",      # 20.
    #         "C(c1ccccc1)(c2ccccc2)CCc3ccccc3",       # 21.
    #         "c1ccc(cc1)CC(c2ccccc2)",               # 22.
    #         "c1ccccc1CCCc2ccccc2",                  # 23.
    #         "CCc1ccccc1CCCc2ccccc2",                # 24.
    #         "c1ccccc1C(C)CCc2ccccc2",               # 25.
    #         "c1ccccc1C(C)(C)CCc2ccccc2",            # 26.
    #         "C(c1ccccc1)(c2ccccc2)C(C)Cc3ccccc3",    # 27.
    #         "c1ccc(cc1)C(C)Cc2ccccc2",              # 28.
    #         "CCc1ccccc1C(C)Cc2ccccc2",              # 29.
    #         "c1ccccc1C(C)(C)C(c2ccccc2)"            # 30.
    #     ],
    #     "Fluoro": [  # 氟基（允许分子中出现多个 –F 或 CF₃ 单元）
    #         # (A) 单氟系列
    #         "CF",                                 # 1. 氟甲烷
    #         "CCF",                                # 2. 氟乙烷
    #         "CCCF",                               # 3. 氟丙烷
    #         "CCCCF",                              # 4. 氟丁烷
    #         "CCCCC(F)",                           # 5. 氟戊烷
    #         "CCCCCCF",                            # 6. 氟己烷
    #         "CCCCCCCF",                           # 7. 氟庚烷
    #         "CCCCCCCCF",                          # 8. 氟辛烷
    #         "CCCCCCCCCF",                         # 9. 氟壬烷
    #         "CCCCCCCCCCF",                        # 10. 氟癸烷
    #         # (B) 三氟甲基系列
    #         "FC(F)(F)",                           # 11. 三氟甲烷
    #         "CC(F)(F)F",                          # 12. 1,1,1-三氟乙烷
    #         "CCC(F)(F)F",                         # 13. 1,1,1-三氟丙烷
    #         "CCCC(F)(F)F",                        # 14. 1,1,1-三氟丁烷
    #         "CCCCC(F)(F)F",                       # 15. 1,1,1-三氟戊烷
    #         "CCCCCC(F)(F)F",                      # 16. 1,1,1-三氟己烷
    #         "C(C(F)(F)F)C(F)(F)F",                 # 17. 六氟乙烷（CF₃–CF₃）
    #         # (C) 双氟系列（在不同碳上各带一个 F）
    #         "C(F)C(F)",                           # 18. 1,2-二氟乙烷
    #         "CC(F)C(F)",                          # 19. 1,2-二氟丙烷
    #         "CCC(F)C(F)",                         # 20. 1,2-二氟丁烷
    #         "CCCC(F)C(F)",                        # 21. 1,2-二氟戊烷
    #         "C(F)(F)C",                           # 22. 2,2-二氟甲烷
    #         "CC(F)(F)C",                          # 23. 2,2-二氟乙烷
    #         "CCC(F)(F)C",                         # 24. 2,2-二氟丙烷
    #         "C(C(F)F)C",                          # 25. 同上另一种写法
    #         "CC(C(F)F)C",                         # 26. 分枝型二氟
    #         "CCC(C(F)F)C",                        # 27. 分枝型二氟
    #         "C(C(F)(F))CC",                       # 28. 另一分枝型
    #         "CC(C(F)(F))CC",                      # 29. 变体
    #         "CCC(C(F)(F))CC"                      # 30. 变体
    #     ],
    #     "Nitro": [  # 硝基（以 N+[O-] 表示，允许单个或多个硝基取代）
    #         "C[N+](=O)[O-]",                      # 1. 硝基甲烷
    #         "CC[N+](=O)[O-]",                     # 2. 硝基乙烷
    #         "CCC[N+](=O)[O-]",                    # 3. 硝基丙烷
    #         "CCCC[N+](=O)[O-]",                   # 4. 硝基丁烷
    #         "CCCCC[N+](=O)[O-]",                  # 5. 硝基戊烷
    #         "CCCCCC[N+](=O)[O-]",                 # 6. 硝基己烷
    #         "C([N+](=O)[O-])([N+](=O)[O-])C",      # 7. 二硝基甲烷（CH(NO₂)₂）
    #         "CC([N+](=O)[O-])C",                   # 8. 2-硝基丙烷
    #         "CC([N+](=O)[O-])CC",                  # 9. 2-硝基丁烷
    #         "CCC([N+](=O)[O-])C",                  # 10. 3-硝基丁烷
    #         "C([N+](=O)[O-])C([N+](=O)[O-])C",      # 11. 1,1-二硝基乙烷
    #         "CC([N+](=O)[O-])C([N+](=O)[O-])C",     # 12. 1,1-二硝基丙烷
    #         "CCC([N+](=O)[O-])C([N+](=O)[O-])C",    # 13. 1,1-二硝基丁烷
    #         "C([N+](=O)[O-])C([N+](=O)[O-])C([N+](=O)[O-])C",  # 14. 三硝基丁烷
    #         "C[N+](=O)[O-]CC[N+](=O)[O-]",          # 15. 两端硝基乙烷
    #         "CC[N+](=O)[O-]CC[N+](=O)[O-]",         # 16. 两端硝基丁烷
    #         "CCC[N+](=O)[O-]CC[N+](=O)[O-]",        # 17. 两端硝基戊烷
    #         "C[N+](=O)[O-]C(C)[N+](=O)[O-]",        # 18.
    #         "CC(C)[N+](=O)[O-]C",                   # 19.
    #         "C([N+](=O)[O-])C([N+](=O)[O-])C([N+](=O)[O-])",  # 20.
    #         "CC(C)([N+](=O)[O-])C",                # 21.
    #         "CC(C)([N+](=O)[O-])CC",               # 22.
    #         "C([N+](=O)[O-])CC(C)[N+](=O)[O-]",     # 23.
    #         "CC(C)[N+](=O)[O-]CC(C)[N+](=O)[O-]",    # 24.
    #         "C([N+](=O)[O-])C(C)[N+](=O)[O-]C",      # 25.
    #         "CC(C)[N+](=O)[O-]C(C)[N+](=O)[O-]",     # 26.
    #         "C([N+](=O)[O-])CC(C)([N+](=O)[O-])",    # 27.
    #         "CC(C)([N+](=O)[O-])C([N+](=O)[O-])",    # 28.
    #         "C([N+](=O)[O-])C([N+](=O)[O-])CC([N+](=O)[O-])",  # 29.
    #         "C([N+](=O)[O-])C([N+](=O)[O-])C([N+](=O)[O-])C([N+](=O)[O-])"  # 30.
    #     ],
    #     "Cyano": [  # 氰基（–C≡N，用 "#N" 表示，允许出现多个 CN 取代）
    #         "CC#N",                              # 1. 乙腈
    #         "CCC#N",                             # 2. 丙腈
    #         "CCCC#N",                            # 3. 丁腈
    #         "CCCCC#N",                           # 4. 戊腈
    #         "CCCCCC#N",                          # 5. 己腈
    #         "C(C#N)(C#N)",                       # 6. 丙二腈
    #         "CC(C#N)(C#N)",                      # 7. 2,2-二氰丙烷
    #         "CCC(C#N)(C#N)",                     # 8. 2,2-二氰丁烷
    #         "C(C#N)CC(C#N)",                     # 9. 1,4-二氰丁烷
    #         "N#CCCN#N",                          # 10. 1,3-二氰丙烷
    #         "CC(C#N)C(C#N)",                     # 11. 2,3-二氰丁烷
    #         "CCC(C#N)C(C#N)",                    # 12. 2,3-二氰戊烷
    #         "C(C#N)(C#N)C(C#N)(C#N)",             # 13. 四氰取代
    #         "CC(C#N)(C#N)C(C#N)(C#N)",            # 14.
    #         "CCC(C#N)(C#N)C(C#N)(C#N)",           # 15.
    #         "C(C#N)CC#N",                        # 16. 三氰取代
    #         "CC(C#N)CC#N",                       # 17.
    #         "CCC(C#N)CC#N",                      # 18.
    #         "C(C#N)(C#N)C#N",                     # 19. 三氰甲基衍生物
    #         "CC(C#N)(C#N)C#N",                    # 20.
    #         "CCC(C#N)(C#N)C#N",                   # 21.
    #         "C(C#N)C(C#N)",                      # 22. 二氰乙烷
    #         "CC(C#N)C(C#N)",                     # 23.
    #         "CCC(C#N)C(C#N)",                    # 24.
    #         "C(C#N)CC(C#N)C",                    # 25.
    #         "CC(C#N)CC(C#N)C",                   # 26.
    #         "CCC(C#N)CC(C#N)C",                  # 27.
    #         "C(C#N)(C#N)CC(C#N)(C#N)",            # 28.
    #         "CC(C#N)(C#N)CC(C#N)(C#N)",           # 29.
    #         "C(C#N)(C#N)C(C#N)(C#N)C"             # 30.
    #     ],
    #     "CarboxylicAcid": [  # 羧酸根（以 -C(=O)[O-] 表示）
    #         "C(=O)[O-]",                         # 1. 甲酸盐
    #         "CC(=O)[O-]",                        # 2. 乙酸盐
    #         "CCC(=O)[O-]",                       # 3. 丙酸盐
    #         "CCCC(=O)[O-]",                      # 4. 丁酸盐
    #         "CCCCC(=O)[O-]",                     # 5. 戊酸盐
    #         "CCCCCC(=O)[O-]",                    # 6. 己酸盐
    #         "C(C(=O)[O-])(C(=O)[O-])",            # 7. 丙二酸盐
    #         "CC(C(=O)[O-])(C(=O)[O-])",           # 8. 2,2-二羧基丙烷
    #         "CCC(C(=O)[O-])(C(=O)[O-])",          # 9. 2,2-二羧基丁烷
    #         "C(=O)[O-]CCC(=O)[O-]",               # 10. 顺丁二酸盐
    #         "CC(=O)[O-]CCC(=O)[O-]",              # 11.
    #         "CCC(=O)[O-]CCC(=O)[O-]",             # 12.
    #         "C(=O)[O-]CCCC(=O)[O-]",              # 13. 己二酸盐
    #         "CC(=O)[O-]CCCC(=O)[O-]",             # 14.
    #         "CCC(=O)[O-]CCCC(=O)[O-]",            # 15.
    #         "C(C(=O)[O-])(C(=O)[O-])C(=O)[O-]",   # 16. 三羧酸盐
    #         "CC(C(=O)[O-])(C(=O)[O-])C(=O)[O-]",  # 17.
    #         "CCC(C(=O)[O-])(C(=O)[O-])C(=O)[O-]", # 18.
    #         "C(=O)[O-]CC(C(=O)[O-])(C(=O)[O-])",   # 19.
    #         "C(=O)[O-]C(C(=O)[O-])(C(=O)[O-])C(=O)[O-]",  # 20.
    #         "CC(=O)[O-]C(C(=O)[O-])(C(=O)[O-])C(=O)[O-]", # 21.
    #         "CCC(=O)[O-]C(C(=O)[O-])(C(=O)[O-])C(=O)[O-]",# 22.
    #         "C(=O)[O-]CC(=O)[O-]",                # 23. 草酸盐
    #         "C(=O)[O-]C(=O)[O-]",                 # 24. 草酸
    #         "CC(=O)[O-]C(=O)[O-]",                # 25.
    #         "C(=O)[O-]CCC(=O)[O-]C(=O)[O-]",       # 26.
    #         "CC(=O)[O-]CC(=O)[O-]CC(=O)[O-]",      # 27.
    #         "C(=O)[O-]CCCC(=O)[O-]C(=O)[O-]",      # 28.
    #         "CC(=O)[O-]CCCC(=O)[O-]C(=O)[O-]",     # 29.
    #         "C(=O)[O-]CCCC(=O)[O-]C(=O)[O-]"        # 30.
    #     ],
    #     "Alkene": [  # 烯（含 C=C 双键，仅限烯这一官能团）
    #         "C=C",                                # 1. 乙烯
    #         "CC=C",                               # 2. 丙烯
    #         "C=C(C)C",                            # 3. 异丁烯
    #         "C=CC=C",                             # 4. 1,3-丁二烯
    #         "C/C=C\\C",                           # 5. trans-2-丁烯
    #         "C\\C=C/C",                           # 6. cis-2-丁烯
    #         "C/C=C\\CC",                          # 7. trans-2-戊烯
    #         "CC/C=C\\C",                          # 8. trans-2-戊烯变体
    #         "C=CC=CC",                            # 9. 1,3-戊二烯
    #         "C=CC=CCC",                           # 10. 1,3-己二烯
    #         "CC=CC=CC",                           # 11. 2-己烯
    #         "C=CC=C(C)C",                         # 12. 带甲基的丁二烯
    #         "C=CC(=C)C",                          # 13. 异构烯
    #         "C(C)=C(C)C",                         # 14. 2,3-二甲基-1-丁烯
    #         "C=CC(=C)C(C)=C",                      # 15. 多烯（含两个双键）
    #         "C=CC(=C)C",                          # 16. 同上简化
    #         "C/C=C\\C/C=C\\C",                     # 17. trans,trans-1,3,5-己三烯
    #         "C/C=C\\C/C=C\\C/C=C\\C",               # 18. trans,trans,trans-1,3,5,7-辛四烯
    #         "CC=CCC=CC",                          # 19. 烯链延长
    #         "C=CCCC=C",                           # 20. 1,5-己二烯
    #         "CC=CCCC=C",                          # 21. 2-己二烯变体
    #         "C=CCCCCC=C",                         # 22. 1,7-庚二烯
    #         "CC=CCCCCC=C",                        # 23. 2-庚二烯变体
    #         "C=CC=CC=CC",                         # 24. 1,3,5-己三烯
    #         "C=CC=CCC=CC",                        # 25. 1,3,6-己三烯
    #         "C/C=C\\C/C=C\\C/C=C\\C",               # 26. 连续多烯链
    #         "CC(C)=C(C)C",                        # 27. 内含双键的分枝烯
    #         "C(C)=C(C)C=C(C)C",                    # 28. 分枝多烯
    #         "CC(C)=C(C)C=C(C)(C)C",                # 29. 更复杂分枝型
    #         "C=CC=CC=CC=CC"                        # 30. 1,3,5,7-庚四烯
    #     ],
    #     "Alkyne": [  # 炔（含 C≡C 三键）
    #         "C#C",                                # 1. 乙炔
    #         "CC#C",                               # 2. 丙炔
    #         "CCC#C",                              # 3. 1-丁炔
    #         "CC#CC",                              # 4. 2-丁炔
    #         "CCCC#C",                             # 5. 1-戊炔
    #         "CC#CCC",                             # 6. 2-戊炔
    #         "CCC#CCC",                            # 7. 1-己炔
    #         "CC#CCCC",                            # 8. 2-己炔
    #         "C#C(C)C",                           # 9. 带甲基的炔
    #         "CC#C(C)C",                          # 10. 2-甲基-1-丁炔
    #         "C#C(C)(C)C",                        # 11. 2,2-二甲基-1-丁炔
    #         "C(C#C)C(C)C",                        # 12. 3,3-二甲基-1-丁炔
    #         "CC#CCC",                             # 13. 1-己炔（重复结构略变）
    #         "C#CC#C",                             # 14. 二炔（如二乙炔）
    #         "CC#CC#C",                            # 15. 2,3-二炔
    #         "C#CC#CC",                            # 16. 连续两个三键
    #         "CC#CC#CC",                           # 17. 延伸二炔
    #         "C#C(C)C#C",                         # 18. 带甲基的二炔
    #         "CC#C(C)C#C",                        # 19. 变体
    #         "C#C(C)C#C(C)C"                       # 20. 多取代二炔
    #     ],
    #     "Alkane": [  # 炔（含 C≡C 三键）
    #         "CC",
    #         "CCC",
    #         "CCCC",
    #         "CCCCC",
    #         "CCCCCC",
    #         "C(C)CC",
    #         "CC(C)CC",          # 异戊烷 (2-甲基丁烷)
    #         "CC(C)(C)C",        # 新戊烷 (异新戊烷)
    #         "CC(C)CCCC",        # 2-甲基己烷
    #         "CCC(C)CCC"         # 3-甲基己烷
    #     ],
    #     "Hydroxy": [
    #         "CO",              # Methanol
    #         "CCO",             # Ethanol
    #         "CCCO",            # 1-Propanol
    #         "CC(O)C",          # 2-Propanol
    #         "CCCCO",           # 1-Butanol
    #         "CC(O)CC",         # 2-Butanol
    #         "CC(C)CO",         # Isobutanol
    #         "CC(C)(C)O",       # tert-Butanol
    #         "CCCCCO",          # 1-Pentanol
    #         "CC(O)CCC",        # 2-Pentanol
    #         "CCC(O)CC",        # 3-Pentanol
    #         "CCCCCCO",         # 1-Hexanol
    #         "CC(O)CCCC",       # 2-Hexanol
    #         "CCC(O)CCC",       # 3-Hexanol
    #         "CCCCCCCO",        # 1-Heptanol
    #         "CCC(O)CCCC",      # 2-Heptanol
    #         "CCCC(O)CCC",      # 3-Heptanol
    #         "OCCO",            # Ethylene glycol
    #         "CC(O)CO",         # Propylene glycol (1,2-propanediol)
    #         "OCCCO",           # 1,3-Propanediol
    #         "OCCCCO",          # 1,4-Butanediol
    #         "OCCCCC(O)",       # 1,5-Pentanediol
    #         "OCCCCCC(O)",      # 1,6-Hexanediol
    #         "OCC(O)CO",        # Glycerol (1,2,3-propanetriol)
    #         "OCC(O)C(O)CO",    # Erythritol
    #         "OCC(O)C(O)C(O)CO",# Xylitol
    #         "OCC(O)C(O)C(O)C(O)CO", # Sorbitol (简化线性表达)
    #         "OC1C(O)C(O)C(O)C(O)C1O", # myo-Inositol (环状六醇)
    #         "OCC(O)C(O)C",     # 1,2,3-Butanetriol (1,2,3-butanetriol)
    #         "OCC(O)C(O)CO"     # 1,2,3,4-Butanetetrol (1,2,3,4-butanetetrol)
    #     ]
    # }
    molecules = {
        "Fluoro": ["CCCCCCCCF",
                    "CCCCCCC(C)F",
                    "CCCCCC(F)CC",
                    "CCCCC(F)CCC",
                    "CCCCCCC(F)CF",
                    "CCCCCC(F)CCF",
                    "CCCCC(F)CCCF",
                    "CCCC(F)CCCCF",
                    "CCC(F)CCCCCF",
                    "CC(F)CCCCCCF",
                    "FCCCCCCCCF",
                    "CCCCCC(F)C(C)F",
                    "CCCCC(F)CC(C)F",
                    "CCCC(F)CCC(C)F",
                    "CCC(F)CCCC(C)F",
                    "CC(F)CCCCC(C)F",
                    "CCCCC(F)C(F)CC",
                    "CCCC(F)CC(F)CC",
                    "CCC(F)CCC(F)CC",
                    "CCCC(F)C(F)CCC",
                    "CCCCCC(F)C(F)CF",
                    "CCCCC(F)CC(F)CF",
                    "CCCC(F)CCC(F)CF",
                    "CCC(F)CCCC(F)CF",
                    "CC(F)CCCCC(F)CF",
                    "FCCCCCCC(F)CF",
                    "CCCCC(F)C(F)CCF",
                    "CCCC(F)CC(F)CCF",
                    "CCC(F)CCC(F)CCF",
                    "CC(F)CCCC(F)CCF",
                    "FCCCCCC(F)CCF",
                    "CCCC(F)C(F)CCCF",
                    "CCC(F)CC(F)CCCF",
                    "CC(F)CCC(F)CCCF",
                    "FCCCCC(F)CCCF",
                    "CCC(F)C(F)CCCCF",
                    "CC(F)CC(F)CCCCF",
                    "CC(F)C(F)CCCCCF",
                    "CCCCC(F)C(F)C(C)F",
                    "CCCC(F)CC(F)C(C)F",
                    "CCC(F)CCC(F)C(C)F",
                    "CC(F)CCCC(F)C(C)F",
                    "CCCC(F)C(F)CC(C)F",
                    "CCC(F)CC(F)CC(C)F",
                    "CC(F)CCC(F)CC(C)F",
                    "CCC(F)C(F)CCC(C)F",
                    "CCCC(F)C(F)C(F)CC",
                    "CCC(F)CC(F)C(F)CC",
                    "CCCCC(F)C(F)C(F)CF",
                    "CCCC(F)CC(F)C(F)CF",],
        "Hydroxy": ["CCCCCCCCO",
                    "CCCCCCC(C)O",
                    "CCCCCC(O)CC",
                    "CCCCC(O)CCC",
                    "CCCCCCC(O)CO",
                    "CCCCCC(O)CCO",
                    "CCCCC(O)CCCO",
                    "CCCC(O)CCCCO",
                    "CCC(O)CCCCCO",
                    "CC(O)CCCCCCO",
                    "OCCCCCCCCO",
                    "CCCCCC(O)C(C)O",
                    "CCCCC(O)CC(C)O",
                    "CCCC(O)CCC(C)O",
                    "CCC(O)CCCC(C)O",
                    "CC(O)CCCCC(C)O",
                    "CCCCC(O)C(O)CC",
                    "CCCC(O)CC(O)CC",
                    "CCC(O)CCC(O)CC",
                    "CCCC(O)C(O)CCC",
                    "CCCCCC(O)C(O)CO",
                    "CCCCC(O)CC(O)CO",
                    "CCCC(O)CCC(O)CO",
                    "CCC(O)CCCC(O)CO",
                    "CC(O)CCCCC(O)CO",
                    "OCCCCCCC(O)CO",
                    "CCCCC(O)C(O)CCO",
                    "CCCC(O)CC(O)CCO",
                    "CCC(O)CCC(O)CCO",
                    "CC(O)CCCC(O)CCO",
                    "OCCCCCC(O)CCO",
                    "CCCC(O)C(O)CCCO",
                    "CCC(O)CC(O)CCCO",
                    "CC(O)CCC(O)CCCO",
                    "OCCCCC(O)CCCO",
                    "CCC(O)C(O)CCCCO",
                    "CC(O)CC(O)CCCCO",
                    "CC(O)C(O)CCCCCO",
                    "CCCCC(O)C(O)C(C)O",
                    "CCCC(O)CC(O)C(C)O",
                    "CCC(O)CCC(O)C(C)O",
                    "CC(O)CCCC(O)C(C)O",
                    "CCCC(O)C(O)CC(C)O",
                    "CCC(O)CC(O)CC(C)O",
                    "CC(O)CCC(O)CC(C)O",
                    "CCC(O)C(O)CCC(C)O",
                    "CCCC(O)C(O)C(O)CC",
                    "CCC(O)CC(O)C(O)CC",
                    "CCCCC(O)C(O)C(O)CO",
                    "CCCC(O)CC(O)C(O)CO",
                    ],
        "Nitro":   ["CCCCCCCC[N+](=O)[O-]",
                    "CCCCCCC(C)[N+](=O)[O-]",
                    "CCCCCC(CC)[N+](=O)[O-]",
                    "CCCCC(CCC)[N+](=O)[O-]",
                    "CCCCCCC(C[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCCCC(CC[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCCC(CCC[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCC(CCCC[N+](=O)[O-])[N+](=O)[O-]",
                    "CCC(CCCCC[N+](=O)[O-])[N+](=O)[O-]",
                    "CC(CCCCCC[N+](=O)[O-])[N+](=O)[O-]",
                    "O=[N+]([O-])CCCCCCCC[N+](=O)[O-]",
                    "CCCCCC(C(C)[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCCC(CC(C)[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCC(CCC(C)[N+](=O)[O-])[N+](=O)[O-]",
                    "CCC(CCCC(C)[N+](=O)[O-])[N+](=O)[O-]",
                    "CC(CCCCC(C)[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCCC(C(CC)[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCC(CC(CC)[N+](=O)[O-])[N+](=O)[O-]",
                    "CCC(CCC(CC)[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCC(C(CCC)[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCCCC(C(C[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCCC(CC(C[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCC(CCC(C[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCC(CCCC(C[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CC(CCCCC(C[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "O=[N+]([O-])CCCCCCC(C[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCCC(C(CC[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCC(CC(CC[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCC(CCC(CC[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CC(CCCC(CC[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "O=[N+]([O-])CCCCCC(CC[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCC(C(CCC[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCC(CC(CCC[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CC(CCC(CCC[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "O=[N+]([O-])CCCCC(CCC[N+](=O)[O-])[N+](=O)[O-]",
                    "CCC(C(CCCC[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CC(CC(CCCC[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CC(C(CCCCC[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCCC(C(C(C)[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCC(CC(C(C)[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCC(CCC(C(C)[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CC(CCCC(C(C)[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCC(C(CC(C)[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCC(CC(CC(C)[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CC(CCC(CC(C)[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCC(C(CCC(C)[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCC(C(C(CC)[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCC(CC(C(CC)[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCCC(C(C(C[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    "CCCC(CC(C(C[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]",
                    ],
        "Cyano":   ["CCCCCCCCC#N",
                    "CCCCCCC(C)C#N",
                    "CCCCCC(C#N)CC",
                    "CCCCC(C#N)CCC",
                    "CCCCCCC(C#N)CC#N",
                    "CCCCCC(C#N)CCC#N",
                    "CCCCC(C#N)CCCC#N",
                    "CCCC(C#N)CCCCC#N",
                    "CCC(C#N)CCCCCC#N",
                    "CC(C#N)CCCCCCC#N",
                    "N#CCCCCCCCCC#N",
                    "CCCCCC(C#N)C(C)C#N",
                    "CCCCC(C#N)CC(C)C#N",
                    "CCCC(C#N)CCC(C)C#N",
                    "CCC(C#N)CCCC(C)C#N",
                    "CC(C#N)CCCCC(C)C#N",
                    "CCCCC(C#N)C(C#N)CC",
                    "CCCC(C#N)CC(C#N)CC",
                    "CCC(C#N)CCC(C#N)CC",
                    "CCCC(C#N)C(C#N)CCC",
                    "CCCCCC(C#N)C(C#N)CC#N",
                    "CCCCC(C#N)CC(C#N)CC#N",
                    "CCCC(C#N)CCC(C#N)CC#N",
                    "CCC(C#N)CCCC(C#N)CC#N",
                    "CC(C#N)CCCCC(C#N)CC#N",
                    "N#CCCCCCCC(C#N)CC#N",
                    "CCCCC(C#N)C(C#N)CCC#N",
                    "CCCC(C#N)CC(C#N)CCC#N",
                    "CCC(C#N)CCC(C#N)CCC#N",
                    "CC(C#N)CCCC(C#N)CCC#N",
                    "N#CCCCCCC(C#N)CCC#N",
                    "CCCC(C#N)C(C#N)CCCC#N",
                    "CCC(C#N)CC(C#N)CCCC#N",
                    "CC(C#N)CCC(C#N)CCCC#N",
                    "N#CCCCCC(C#N)CCCC#N",
                    "CCC(C#N)C(C#N)CCCCC#N",
                    "CC(C#N)CC(C#N)CCCCC#N",
                    "CC(C#N)C(C#N)CCCCCC#N",
                    "CCCCC(C#N)C(C#N)C(C)C#N",
                    "CCCC(C#N)CC(C#N)C(C)C#N",
                    "CCC(C#N)CCC(C#N)C(C)C#N",
                    "CC(C#N)CCCC(C#N)C(C)C#N",
                    "CCCC(C#N)C(C#N)CC(C)C#N",
                    "CCC(C#N)CC(C#N)CC(C)C#N",
                    "CC(C#N)CCC(C#N)CC(C)C#N",
                    "CCC(C#N)C(C#N)CCC(C)C#N",
                    "CCCC(C#N)C(C#N)C(C#N)CC",
                    "CCC(C#N)CC(C#N)C(C#N)CC",
                    "CCCCC(C#N)C(C#N)C(C#N)CC#N",
                    "CCCC(C#N)CC(C#N)C(C#N)CC#N",],
        "Carboxy": ["CCCCCCCCC(=O)O",
                    "CCCCCCC(C)C(=O)O",
                    "CCCCCC(CC)C(=O)O",
                    "CCCCC(CCC)C(=O)O",
                    "CCCCCCC(CC(=O)O)C(=O)O",
                    "CCCCCC(CCC(=O)O)C(=O)O",
                    "CCCCC(CCCC(=O)O)C(=O)O",
                    "CCCC(CCCCC(=O)O)C(=O)O",
                    "CCC(CCCCCC(=O)O)C(=O)O",
                    "CC(CCCCCCC(=O)O)C(=O)O",
                    "O=C(O)CCCCCCCCC(=O)O",
                    "CCCCCC(C(=O)O)C(C)C(=O)O",
                    "CCCCC(CC(C)C(=O)O)C(=O)O",
                    "CCCC(CCC(C)C(=O)O)C(=O)O",
                    "CCC(CCCC(C)C(=O)O)C(=O)O",
                    "CC(CCCCC(C)C(=O)O)C(=O)O",
                    "CCCCC(C(=O)O)C(CC)C(=O)O",
                    "CCCC(CC(CC)C(=O)O)C(=O)O",
                    "CCC(CCC(CC)C(=O)O)C(=O)O",
                    "CCCC(C(=O)O)C(CCC)C(=O)O",
                    "CCCCCC(C(=O)O)C(CC(=O)O)C(=O)O",
                    "CCCCC(CC(CC(=O)O)C(=O)O)C(=O)O",
                    "CCCC(CCC(CC(=O)O)C(=O)O)C(=O)O",
                    "CCC(CCCC(CC(=O)O)C(=O)O)C(=O)O",
                    "CC(CCCCC(CC(=O)O)C(=O)O)C(=O)O",
                    "O=C(O)CCCCCCC(CC(=O)O)C(=O)O",
                    "CCCCC(C(=O)O)C(CCC(=O)O)C(=O)O",
                    "CCCC(CC(CCC(=O)O)C(=O)O)C(=O)O",
                    "CCC(CCC(CCC(=O)O)C(=O)O)C(=O)O",
                    "CC(CCCC(CCC(=O)O)C(=O)O)C(=O)O",
                    "O=C(O)CCCCCC(CCC(=O)O)C(=O)O",
                    "CCCC(C(=O)O)C(CCCC(=O)O)C(=O)O",
                    "CCC(CC(CCCC(=O)O)C(=O)O)C(=O)O",
                    "CC(CCC(CCCC(=O)O)C(=O)O)C(=O)O",
                    "O=C(O)CCCCC(CCCC(=O)O)C(=O)O",
                    "CCC(C(=O)O)C(CCCCC(=O)O)C(=O)O",
                    "CC(CC(CCCCC(=O)O)C(=O)O)C(=O)O",
                    "CC(C(=O)O)C(CCCCCC(=O)O)C(=O)O",
                    "CCCCC(C(=O)O)C(C(=O)O)C(C)C(=O)O",
                    "CCCC(CC(C(=O)O)C(C)C(=O)O)C(=O)O",
                    "CCC(CCC(C(=O)O)C(C)C(=O)O)C(=O)O",
                    "CC(CCCC(C(=O)O)C(C)C(=O)O)C(=O)O",
                    "CCCC(C(=O)O)C(CC(C)C(=O)O)C(=O)O",
                    "CCC(CC(CC(C)C(=O)O)C(=O)O)C(=O)O",
                    "CC(CCC(CC(C)C(=O)O)C(=O)O)C(=O)O",
                    "CCC(C(=O)O)C(CCC(C)C(=O)O)C(=O)O",
                    "CCCC(C(=O)O)C(C(=O)O)C(CC)C(=O)O",
                    "CCC(CC(C(=O)O)C(CC)C(=O)O)C(=O)O",
                    "CCCCC(C(=O)O)C(C(=O)O)C(CC(=O)O)C(=O)O",
                    "CCCC(CC(C(=O)O)C(CC(=O)O)C(=O)O)C(=O)O",]
    }
    # 输出目录
    out_dir = "new_xyz_structures"
    os.makedirs(out_dir, exist_ok=True)

    # 遍历每个官能团和对应的 SMILES 列表
    for group, smiles_list in molecules.items():
        for i, smi in enumerate(smiles_list):
            # 如果是羧酸类别，将 "[O-]" 替换为 "O"（生成中性羧酸）
            if group == "CarboxylicAcid":
                smi = smi.replace("[O-]", "O")
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                print(f"无法解析 SMILES: {smi}")
                continue
            # 添加氢原子
            mol = Chem.AddHs(mol)
            # 生成 3D 构型（使用 ETKDG 算法）
            result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
            if result != 0:
                print(f"嵌入构型失败: {smi}")
                continue
            # UFF 优化几何结构
            AllChem.UFFOptimizeMolecule(mol)
            xyz_content = generate_xyz(mol, comment=group, smiles=smi)
            # 为每个分子生成单独的 xyz 文件，文件名格式为：Group_index.xyz
            filename = os.path.join(out_dir, f"{group}_{i+1:03d}.xyz")
            with open(filename, "w") as f:
                f.write(xyz_content)
            print(f"已保存 {filename}")

if __name__ == "__main__":
    main()
