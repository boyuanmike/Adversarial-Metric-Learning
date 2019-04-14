from models.modifiedgooglenet import ModifiedGoogLeNet


def train(device, dataloader, dataset, params, epochs=20):
    model = ModifiedGoogLeNet(params.out_dims, params.normalize_output).to(device)

    for anchors, positives, negatives, neg_indices in dataloader:
        anchors = anchors.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)
