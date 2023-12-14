import umap
import numpy as np
import matplotlib.pyplot as plt
import torch
rbps = ["ALKBH5_Baltz2012",
"C17ORF85_Baltz2012",
"C22ORF28_Baltz2012",
"CAPRIN1_Baltz2012",
"CLIPSEQ_AGO2",
"CLIPSEQ_ELAVL1",
"CLIPSEQ_SFRS1",
"ICLIP_HNRNPC",
"ICLIP_TDP43",
"ICLIP_TIA1",
"ICLIP_TIAL1",
# "PARCLIP_AGO1234",
"PARCLIP_ELAVL1",
"PARCLIP_ELAVL1A",
"PARCLIP_EWSR1",
"PARCLIP_FUS",
"PARCLIP_HUR",
# "PARCLIP_IGF2BP123",
"PARCLIP_MOV10_Sievers",
"PARCLIP_PUM2",
"PARCLIP_QKI",
"PARCLIP_TAF15",
"PTBv1",
"ZC3H7B_Baltz2012"]
rbp_paths = [
    '/amax/data/gaoyifei/GraphProt/GraphProt_CLIP_sequences/'+ rbp + '/' + rbp +'.pt' for rbp in rbps
]
# 创建一个空列表来存储所有 RBP 的表征数据
rbp_representations = []

# 逐个加载每个 RBP 的表征数据
for rbp_path in rbp_paths:
    try:
        # 加载 RBP 表征数据
        print(rbp_path + ' begins.')
        rbp_embedding = torch.load(rbp_path)['representations'][12]

        if isinstance(rbp_embedding, torch.Tensor):
            mean_protein = torch.mean(rbp_embedding,dim = 0, keepdim=True)
            mean_protein = mean_protein.numpy()
            rbp_representations.append(mean_protein)
    except Exception as e:
        print(f"Error loading {rbp_path}: {str(e)}")

# 将表征数据堆叠成一个大矩阵
X = np.vstack(rbp_representations)

# 使用 UMAP 进行降维
reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(X)

# 创建一个颜色映射，将每个 RBP 映射到一个颜色
color_map = plt.cm.get_cmap("tab20", len(rbps))

# 绘制 UMAP 可视化图
fig = plt.figure(figsize=(12, 8))

for i, rbp_name in enumerate(rbps):
    color = color_map(i)
    plt.scatter(embedding[i, 0], embedding[i, 1], color=color, label=rbp_name)

# 添加标签（可以根据需要添加 RBP 名称作为标签）
for i, rbp_name in enumerate(rbps):
    plt.annotate(rbp_name, (embedding[i, 0], embedding[i, 1]), fontsize=6)

plt.title('UMAP Visualization of RBP Representations')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.show()
plt.savefig('/home/gaoyifei/RNARepresentation/out/superpixels_graph_classification/results/umap_vis.png')

# ax = fig.add_subplot(111, projection='3d')
# for i, rbp_name in enumerate(rbps):
#     color = color_map(i)
#     x, y, z = embedding[i]
#     ax.scatter(x, y, z, c=[color], label=rbp_name)

# for i, rbp_name in enumerate(rbps):
#     x, y, z = embedding[i]
#     ax.text(x, y, z, rbp_name, fontsize=8)

# ax.set_title("3D UMAP Visualization of Protein Representations")
# ax.set_xlabel("UMAP Dimension 1")
# ax.set_ylabel("UMAP Dimension 2")
# ax.set_zlabel("UMAP Dimension 3")
# plt.legend()
# plt.show()
# plt.savefig('/home/gaoyifei/RNARepresentation/out/superpixels_graph_classification/results/umap_vis3.png')
