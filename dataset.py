import os

from torch.utils.data import Dataset


class ChessDataset(Dataset):
    """
    Constructor. This function should take in a file path where the root directory of the dataset lies, as long
    with some transforms that are to be done on the data.
    TODO: Implement
    """
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        #self.img_labels = pd.read_csv(annotations_file)
        #self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    """
    This function should return the number of data examples in our dataset
    TODO: Implement and test that this works
    """
    def __len__(self):
        pass
        #return len(self.img_labels)

    """
    This function should return an example (data, target) in our dataset given some index (idx). We should also
    apply the transforms on the data and target before returning the example.
    TODO: Implement and test that this works
    """
    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        # image = read_image(img_path)
        # label = self.img_labels.iloc[idx, 1]
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label
        pass