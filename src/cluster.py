import networkx as nx
import numpy as np


def cluster_boxes(boxes):
    graph = nx.Graph()
    for i in range(boxes.shape[0]):
        ioverlap = np.where(np.logical_and(
            np.all(boxes[i, :2] <= boxes[i+1:, 2:], axis=1),
            np.all(boxes[i, 2:] >= boxes[i+1:, :2], axis=1)))[0] + i + 1
        if ioverlap.size > 0:
            ebunch = np.stack(
                [i * np.ones(ioverlap.size, dtype=np.int), ioverlap]).T
            graph.add_edges_from(ebunch)
        else:
            graph.add_node(i)

    return nx.connected_components(graph)


if __name__ == '__main__':
    boxes = np.array([[0, 0, 5, 6],
        [1, 1, 3, 3], 
        [2, 2, 4, 4], 
        [4, 5, 6, 7],
        [8, 1, 10, 4],
        [9, 3, 11, 5],
        [11, 2, 13, 4]])
    clusters = list(cluster_boxes(boxes))
    print(clusters)

    boxes = np.array([[14.79661574, 21.21601612, 15.30176893, 21.72116931],
        [18.48462587, 7.41137572, 19.03767023, 7.96442009],
        [2.69673224, 9.59842375, 3.32057219, 10.22226371]])
    clusters = list(cluster_boxes(boxes))
    print(clusters)

    boxes = np.array([[6.81938108, 18.65041178, 7.52721282, 19.35824351],
        [10.47957221,  4.74881386, 10.97182461,  5.24106625],
        [13.85465225, 10.67964592, 14.85465225, 11.67964592],
        [ 7.64280886, 18.18693742,  8.34956098, 18.89368953],
        [ 8.59207879, 18.34769795,  9.21301121, 18.96863037],
        [12.26176147,  4.42352969, 12.86341253,  5.02518075],
        [ 9.15      , 18.63943225,  9.75556467, 19.24499692],
        [14.10533103,  5.69170303, 14.68470125,  6.27107324],
        [ 9.97261293, 19.30506776, 10.55050568, 19.88296051],
        [10.80585378, 19.7812637 , 11.28636705, 20.26177698],
        [24.60318075, 11.85      , 25.01565066, 12.26246991],
        [14.78231664,  6.11960893, 15.29557712,  6.63286942],
        [11.95292709, 20.39982668, 12.55003767, 20.99693726],
        [17.46145317,  7.17423254, 17.89978823,  7.6125676 ],
        [14.79661574, 21.21601612, 15.30176893, 21.72116931],
        [18.48462587,  7.41137572, 19.03767023,  7.96442009],
        [ 2.69673224,  9.59842375,  3.32057219, 10.22226371]])
    
    import matplotlib.pyplot as plt
    import matplotlib.patches as pat
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    for b in boxes:
        ax.add_patch(pat.Rectangle(b[:2], b[2]-b[0], b[3]-b[1]))
    plt.xlim([0, 30])
    plt.ylim([0, 30])
    plt.show()
    clusters = list(cluster_boxes(boxes))
    print(clusters)
