import os
import sys

sys.path.append('./sample')
import json

import cv2
import numpy as np
from PIL import Image


class SampleThread():
    def __init__(self, parent, source, mode, topk, num_samples):
        super(SampleThread, self).__init__()
        self.parent = parent
        self.source = source
        self.mode = mode
        self.topk = topk
        self.img_size = self.parent.img_size.copy()
        self.num_samples = num_samples
    
    def run(self):
        os.makedirs(os.path.join('demo', 'results'), exist_ok=True)
        os.makedirs(os.path.join('demo', 'results', 'stroke_cond'), exist_ok=True)
        os.makedirs(os.path.join('demo', 'results', 'text_cond'), exist_ok=True)
        os.makedirs(os.path.join('demo', 'results', 'exemplar_cond'), exist_ok=True)

        I_gray = self.parent.input_image.copy()
        I_gray.save(os.path.join('demo', 'results', 'gray.png'))

        all_strokes = []

        if 'stroke' in self.mode:
            if self.source == 'input':
                strokes = self.parent.input_strokes.get_strokes()
            elif self.source == 'output':
                strokes = self.parent.output_strokes.get_strokes()
            all_strokes += strokes
            (self.parent.get_pixmap_image(self.source)).save(os.path.join('demo', 'results', 'stroke_cond', 'strokes.png'))
            save_strokes(I_gray, strokes, self.img_size, os.path.join('demo', 'results', 'stroke_cond'))

        # Delete duplicate strokes
        strokes = []
        for stk in all_strokes:
            for s in strokes:
                if stk['index'] == s['index']:
                    break
            else:
                strokes.append(stk)

        save_strokes(I_gray, strokes, self.img_size, os.path.join('demo', 'results'))

        gen_imgs = []
        if self.source == 'input':
            for i in range(self.num_samples):
                self.message.emit(f'Sampling... {i+1}/{self.num_samples}')
                gen = self.parent.colorizer.sample(I_gray, strokes, self.topk, progress=self.progress)
                gen.save(os.path.join('demo', 'results', f'colorized_{i}.png'))
                gen_imgs.append(gen)

        self.parent.sample_btn.setEnabled(True)

        for img in gen_imgs:
            self.result.emit(img, False)

def draw_full_color(l, color, rect):
    y0, y1, x0, x1 = rect
    img = np.array(l.convert('RGB'))
    draw = np.array(color).astype(np.uint8)
    if len(draw.shape) == 1:
        draw = np.expand_dims(draw, axis=[0, 1])
    img[y0:y1, x0:x1, :] = draw[:, :, :]
    return Image.fromarray(img)

def draw_strokes(image, img_size, strokes):
    org_size = image.size[::-1]
    # Draw strokes
    draw_img = image.copy().convert('RGB')
    for stk in strokes:
        ind = stk['index']
        ind = [int(ind[0] / img_size[0] * org_size[0]), int(ind[1] / img_size[1] * org_size[1])]
        patch_size = [int(org_size[0]/img_size[0]*10), int(org_size[1]/img_size[1]*10)]
        border_size = [int(org_size[0]/img_size[0]*3), int(org_size[1]/img_size[1]*3)]
        color = np.zeros(patch_size+[3])
        color[:, :] = stk['color']
        color = color.astype(np.uint8)
        color = cv2.copyMakeBorder(color, border_size[0], border_size[0], border_size[1], border_size[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
        draw_img = draw_full_color(draw_img, color, [ind[0], ind[0]+color.shape[0], ind[1], ind[1]+color.shape[1]])
    return draw_img.resize(org_size[::-1])
    

def save_strokes(I_gray, strokes, img_size, path):
    for stk in strokes:
        stk['index'] = np.int32(stk['index']).tolist()
        stk['color'] = np.int32(stk['color']).tolist()
    draw_img = draw_strokes(I_gray, img_size, strokes)
    draw_img.save(os.path.join(path, 'points.png'))
    with open(os.path.join(path, 'strokes.json'), 'w') as f:
        json.dump(strokes, f)