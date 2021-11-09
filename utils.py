import torch
import tqdm
import cv2
import time
import numpy as np

def draw_caption(image, box, caption):
	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def visualize(cfg, model, dl_ev):
    model.eval()
    model.training = False
    
    pbar = tqdm.tqdm(enumerate(dl_ev))
    for idx, data in pbar:
        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = model(data['image'].cuda().float())
            else:
                scores, classification, transformed_anchors = model(data['image'].float())
            pbar.set_description('Elapsed time: {}'.format(time.time()-st))
            idxs = np.where(scores.cpu()>0.95)
            img = np.array(255 * data['image'][0, :, :, :]).copy()

            img[img<0] = 0
            img[img>255] = 255

            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                draw_caption(img, (x1, y1, x2, y2),
                             str(classification[idxs[0][j]].item())+"-"+str(scores[idxs[0][j]].item())[0:4])
                
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            cv2.imwrite("./inference_examples/{}.jpg".format(idx), img)
    
    model.train()
    model.training = True
    
    