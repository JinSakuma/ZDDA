import numpy as np
import matplotlib.pyplot as plt
import umap


def get_feat(cnn1, cnn2, loader, device, target_dict, N):
    feat_list1 = []
    feat_list2 = []
    for batch1, batch2 in loader:
        X1, y1, d1 = batch1
        X2, y2, d2 = batch1
        X1, y1 = X1.to(device), y1.to(device)
        X2, y2 = X2.to(device), y2.to(device)
    
        if target_dict[str(int(y1.detach().cpu().numpy()))]>N-1:
            continue
        else:
            target_dict[str(int(y1.detach().cpu().numpy()))] += 1

            # feature extraction
            feat1 = cnn1(X1)
            feat2 = cnn2(X1)

            feat1 = feat1.view(-1, 256*6*6)
            feat2 = feat2.view(-1, 256*6*6)

            feat_list1.append(feat1[0].detach().cpu().numpy())
            feat_list2.append(feat2[0].detach().cpu().numpy())
            
    return feat_list1, feat_list2


def show_UMAP_2D(featAB, featABCD, filepath=None, random_seed=0, cls=5, S=10):
    reducer = umap.UMAP(n_components=2, random_state=random_seed, n_neighbors=5)
    reducer.fit(featAB)
    umap_vecs = reducer.transform(featABCD)
    A_vecs = umap_vecs[:S*cls]
    B_vecs = umap_vecs[S*cls:S*cls*2]
    C_vecs = umap_vecs[S*cls*2:S*cls*3]
    D_vecs = umap_vecs[S*cls*3:S*cls*4]
    plt.figure(figsize=(10, 10))
    for i in range(cls):
        plt.scatter(A_vecs[i*S:(i+1)*S, 0], A_vecs[i*S:(i+1)*S, 1], c='r', marker="${}$".format(str(i)))
        plt.scatter(B_vecs[i*S:(i+1)*S, 0], B_vecs[i*S:(i+1)*S, 1], c='b', marker="${}$".format(str(i+cls)))
        plt.scatter(C_vecs[i*S:(i+1)*S, 0], C_vecs[i*S:(i+1)*S, 1], c='g', marker="${}$".format(str(i)))
        plt.scatter(D_vecs[i*S:(i+1)*S, 0], D_vecs[i*S:(i+1)*S, 1], c='k', marker="${}$".format(str(i+cls)))
            
    if filepath is not None:
        plt.savefig(filepath)
#     plt.show()