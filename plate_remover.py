from keras.models import load_model
import os.path as osp
from PIL import Image
import numpy as np
from datetime import datetime
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm

from moviepy.editor import VideoFileClip, ImageSequenceClip


class PlateRemover():
    def __init__(self, model_file, output_location, color, vsplits, hsplits):
        self.model = load_model(model_file)
        self.output_loc = output_location
        self.color = color
        self.vsplits = vsplits
        self.hsplits = hsplits

    @staticmethod
    def is_vertical(rotation):
        return rotation // 90 % 2 != 0

    @staticmethod
    def timestamp():
        return datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    @staticmethod
    def split(img_np, split_v, split_h):
        cols = np.array_split(img_np, split_h, axis=1)
        rows = [np.array_split(c, split_v, axis=0) for c in cols]
        return rows

    @staticmethod
    def merge(n_imgs):
        return np.hstack(list((np.vstack(tuple(col)) for col in n_imgs)))

    def get_prediction(self, image):
        h, w = self.model.layers[0].output_shape[1:3]
        img_feed = np.expand_dims(np.asarray(image.resize((w,h), resample=Image.BILINEAR)), axis=0)/255.
        return self.model.predict(img_feed)[0,:,:,:]

    def wipe_from_image(self, image, threshold, alpha):
        result = image.copy()

        pred = self.get_prediction(image)

        depth = list(self.color)
        depth[3] *= alpha

        if self.vsplits > 1 or self.hsplits > 1:

            v_splits, h_splits = self.vsplits, self.hsplits;                                            # splits
            img_np = np.asarray(image);                                                                 # splits
            splits = self.split(img_np, v_splits, h_splits);                                            # splits
            split_preds = [[self.get_prediction(Image.fromarray(i)) for i in col] for col in splits]    # splits
            merged_preds = self.merge(split_preds)                                                      # splits

            if threshold:
                merged_preds[merged_preds < threshold] = 0.                                             # splits
            if len(merged_preds[merged_preds > 0]) > 0:                                                 # splits
                minmax_scale(merged_preds[merged_preds > 0], copy=False)                                # splits

            mask_split = Image.fromarray(np.uint8(np.dot(merged_preds, np.array([depth]) )))            # splits
            mask_split = mask_split.resize(image.size, resample=Image.BILINEAR)                         # splits

            result.paste(mask_split, mask=mask_split)                                                         # splits

        if threshold:
            pred[pred < threshold] = 0.
        if len(pred[pred>0]) > 0:
            minmax_scale(pred[pred > 0], copy=False)

        mask = Image.fromarray(np.uint8(np.dot(pred, np.array([depth]) )))
        mask = mask.resize(image.size, resample=Image.BILINEAR)

        result.paste(mask, mask=mask)

        return result


    def wipe_from_video(self, src_clip, threshold, alpha, keep_audio):

        resultFrames = []

        for frame in tqdm(src_clip.iter_frames()):
            dst  = self.wipe_from_image(Image.fromarray(frame),
                                        threshold=threshold,
                                        alpha=alpha)
            resultFrames.append(np.asarray(dst))

        resultClip = ImageSequenceClip(resultFrames, fps=src_clip.fps,
                                       with_mask=False)
        if keep_audio:
            resultClip = resultClip.set_audio(src_clip.audio)

        return resultClip

    def remove_plates(self, filename, keep_audio=True, alpha=1):
        clip = VideoFileClip(filename)

        if clip.w > clip.h and self.is_vertical(clip.rotation):
            print("opa zhopa")
            clip = VideoFileClip(filename, target_resolution=(clip.w, clip.h))

        wiped_clip = self.wipe_from_video(clip, threshold=.3, alpha=alpha,
                                          keep_audio=keep_audio)

        target_name = osp.join(self.output_loc, '{}.mp4'.format(self.timestamp()))
        wiped_clip.write_videofile(target_name, audio_codec='aac', remove_temp=True)
        return target_name
