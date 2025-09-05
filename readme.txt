note:*表示主干训练过程的部分
所有的训练以及微调脚本均使用args，如果需要试用可以直接进入查看具体用法

*charge_predict中的是使用EGNN，以原子index和坐标为输入，输出电荷的一系列脚本，预测得到的模型是预训练模型
	comenet4charge.py和features.py文件是预训练阶段的架构文件
	training_charge_model.py是训练用的脚本，主要用于调整不同的维度、训练集等
	dataset_without_charge.py是用于提取QM9数据集的dataloader脚本

*fine-tune里的脚本是对charge_predict生成的预训练模型进行微调，pooling得到单一性质的模型
	预训练阶段用到的架构文件均需要用到
	FineTunedModel.py和embedding2property.py调用了comenet4charge.py中的预训练架构，并接入了微调时的层
	fine_tune_training.py是用于微调的脚本，在使用时必须使用预训练已经训练好的模型，我们主要使用的模型是2024/12/7 训练得到的best_model_dim128_1.pth

benchmark里是测试公开数据集的结果，其中network为微调脚本所在处，dataset为储存有几何结构的数据集所在处
	使用的微调脚本和fine-tune中基本一致，除了dataloader更换为了适用于微调数据集的形式
	benchmark目录下的py文件是可以生成标准xyz格式的脚本和处理测试集中带有无法解析原子分子的脚本

data是QM9的数据集的文件夹，里面有几个不同尺寸的训练集

下面是比较和画图部分

summary/中存放着生成figure2和figure3的脚本
	其中GNN_EGNN_comparison.py是GNN和EGNN在预训练上的准确度比较的散点图生成脚本，使用方法为python GNN_EGNN_comparison.py --checkpoint_a  best_gnn_transformer_model_1.pth --checkpoint_b best_model_dim128_1.pth --model_a smiles2vec --model_b comenet_charge --test_data_root ../../data/data_valid_test/ --batch_size 64 --plot_path charge_compare.svg
	finetune_vs_direct是用微调模型和直接训练模型在QM9测试集的不同样本量和不同物理量上得到的结果比较的折线图生成脚本，使用方法为python finetune_vs_direct.py --log-dir finetune_direct_test/ --out finetune_vs_direct_r2.svg
	property_predict_qm7是微调模型和直接训练模型在QM7测试集上的准确度比较的散点图生成脚本，使用方法为python property_predict_qm7.py --checkpoint_a ../result/QM7/finetune_direct_test/best_qm7_layer_4_data500.pth --checkpoint_b ../result/QM7/finetune_direct_test/best_qm7_layer_0_data_500.pth --test_data_root ../data/qm7_divide/test/ --target_property rot_A --plot_path qm7.svg
	atom_embedding_dim是不同潜空间维度和预训练样本量下的结果比较的折线图生成脚本，使用方法为python atom_embedding_dim.py --root atom_dim_test/
	auto_freeze_layer是消融实验的数据提取器，使用方法为python auto_freeze_layer.py --log_dir freeze_layer_test/HOMO/ --output freeze_layer.pdf

analysis/中存放着聚类实验所用到的脚本
	embedding_plot是用于我们手工构建数据集生成聚类分析结果的脚本，用法是python embedding_plot.py --xyz_dir new_xyz_structures/ --model_path best_model_dim128_1.pth --device cuda --perplexity 12 --batch_size 50
	embedding_plot_qm9是用于我们QM9数据集生成聚类分析结果的脚本，用法是python embedding_plot_qm9.py --xyz_dir ../../database/ --model_path best_model_dim128_1.pth --device cuda --perplexity 12 --batch_size 50   --mw_plot --dipole_plot --binary_group_plots
	生成的图片均在analysis/latent_vis_results中