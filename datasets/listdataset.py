import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data


class ListDataset(data.Dataset):
    def __init__(self, root, dataset, path_list, transform=None, target_transform=None,
                 co_transform=None, loader=None, datatype=None, scale=True, use_MI=True):

        self.root = root
        self.dataset = dataset
        self.img_path_list = path_list
        self.transform = transform
        self.target_transform = target_transform
        self.co_transform = co_transform
        self.loader = loader
        self.datatype = datatype
        self.scale = scale
        self.use_MI = use_MI

    def __getitem__(self, index):
        img_path = self.img_path_list[index][:-1]
        # We do not consider other datsets in this work
        assert self.dataset == 'bsd500'
        assert (self.transform is not None) and (self.target_transform is not None)

        inputs, gtlabel, brlabel,hsv,lab = self.loader(img_path, img_path.replace('_img.jpg', '_sen.png'),
                                               img_path.replace('_img.jpg', '_br.png'))
    

        if self.scale:
            inputs, gtlabel, brlabel,hsv,lab = self.generate_scale_label(inputs, gtlabel, brlabel,hsv,lab )

        if self.co_transform is not None:

           # label = np.concatenate((gtlabel, brlabel), axis=2)
            inputs,  gtlabel, brlabel, hsv,lab = self.co_transform(
                [inputs.astype(np.float32)],
                gtlabel, brlabel, hsv,lab
            )
           # gtlabel, brlabel = np.split(label, 2, axis=-1)

        if self.transform is not None:
            image = self.transform(inputs[0])

        if self.target_transform is not None:
            # print(label.shape)
            br = torch.tensor(brlabel, dtype=torch.float32).permute(2, 0, 1)
            sen = torch.tensor(gtlabel, dtype=torch.float32).permute(2, 0, 1)
            hsv = torch.tensor(hsv, dtype=torch.float32).permute(2, 0, 1)
            lab = torch.tensor(lab, dtype=torch.float32).permute(2, 0, 1)

            gtlabel = self.target_transform(gtlabel[:, :, :1])
            brlabel = self.target_transform(brlabel[:, :, :1])



        if self.use_MI:

            return image, gtlabel,brlabel,sen, br

        else:
            return image, gtlabel, brlabel

    def generate_scale_label(self, image, gtlabel, brlabel,hsv,lab):
        f_scale = 0.7 + np.random.randint(0, 8) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_CUBIC)

        gtlabel = cv2.resize(gtlabel, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        brlabel = cv2.resize(brlabel, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        hsv = cv2.resize( hsv, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        lab = cv2.resize(lab, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)


        return image, gtlabel, brlabel,hsv,lab



    def __len__(self):
        return len(self.img_path_list)



