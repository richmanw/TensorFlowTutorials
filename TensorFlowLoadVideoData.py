import tqdm
import random
import pathlib
import itertools
import collections

import os
import cv2
import numpy as np
import remotezip as rz
import tensorflow as tf
import imageio

from IPython import display
from urllib import request
from tensorflow_docs.vis import embed

def list_files_from_zip_url(zip_url):
    """
    List the files in each class of the dataset given a URL with the zip file.

        Args:
            zip_url: A URL from which the files can be extracted from.

        Returns:
            List of files in each of the classes
    """
    files = []
    with rz.RemoteZip(zip_url) as zip:
        for zip_info in zip.infolist():
            files.append(zip_info.filename)
    return files

def get_class(fname):
    """
    Retrieve the name of the class given a filename.

        Args:
            fname: Name of the file in the UCF101 dataset.

        Returns:
            Class that the file belongs to.
    """
    return fname.split('_')[-3]

def get_files_per_class(files):
    """
    Retrieve the files that belong to each class.

        Args:
            files: List of files in the dataset.

        Returns:
            Dictionary of class names (key) and files (values).
    """
    files_for_class = collections.defaultdict(list)
    for fname in files:
        class_name = get_class(fname)
        files_for_class[class_name].append(fname)
    return files_for_class

def select_subset_of_classes(files_for_class, classes, files_per_class):
    """
    Creates a dictionary with the class name and a subset of the fields in that class.

        Args:
            files_for_class: Dictionary of class names (key) and files (values).
            classes: List of classes.
            files_per_class: Number of files per class of interest.

        Returns:
            Dictionary with class as key and list of specified number of video files in that class.
    """

    files_subset = dict()

    for class_name in classes:
        class_files = files_for_class[class_name]
        files_subset[class_name] = class_files[:files_per_class]

    return files_subset

def download_from_zip(zip_url, to_dir, file_names):
    """
    Download the contents of the zip file from the zip URL.

        Args:
            zip_url: A URL with a zip file containing data.
            to_dir: A directory to download data to.
            file_names: Names of files to download.

        Returns:
            Nothing.
    """
    with rz.RemoteZip(zip_url) as zip:
        for fn in tqdm.tqdm(file_names):
            class_name = get_class(fn)
            zip.extract(fn, str(to_dir / class_name))
            unzipped_file = to_dir / class_name / fn

            fn = pathlib.Path(fn).parts[-1]
            output_file = to_dir / class_name / fn
            unzipped_file.rename(output_file)

def split_class_lists(files_for_class, count):
    """
    Returns the list of files belonging to a subset of data as well as the
    remainder of files that need to be downloaded.

        Args:
            files_for_class: Files belonging to a particular class of data.
            count: Number of files to download.

        Returns:
            Files belonging to the subset of data and dictionary of the
            remainder of files that need to be downloaded
    """
        # LIST
    split_files = []

        # DICTIONARY
    remainder = {}

    for cls in files_for_class:
        split_files.extend(files_for_class[cls][:count])
        remainder[cls] = files_for_class[cls][count:]
    return split_files, remainder

def download_ucf_101_subset(zip_url, num_classes, splits, download_dir):
    """
    Download a subset of the UCF101 dataset and split them into various parts,
    such as training, validation, and testing

        Args:
            zip_url: A URL with a ZIP file with data.
            num_classes: Number of labels.
            splits: Dictionary specifying the training, validation, testing,
            etc. (key) division of data (value is number of files per split)
            download_dir: Directory to download data to.

        Return:
            Mapping of the directories containing the subsections of data.
    """

    files = list_files_from_zip_url(zip_url)
    for f in files:
        path = os.path.normpath(f)
        tokens = path.split(os.sep)
        if len(tokens) <= 2:
            files.remove(f) # Remove that item from the list if it does not have a file name

    files_for_class = get_files_per_class(files)
    classes = list(files_for_class.keys())[:num_classes]

    for cls in classes:
        random.shuffle(files_for_class[cls])

    # Only use the number of classes you want in the dictionary
    files_for_class = {x: files_for_class[x] for x in classes}

    dirs = {}
    for split_name, split_count in splits.items():
        print(split_name, ":")
        split_dir = download_dir / split_name
        split_files, files_for_class = split_class_lists(files_for_class, split_count)
        download_from_zip(zip_url, split_dir, split_files)
        dirs[split_name] = split_dir

    return dirs

def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

        Args:
             frame: Image that needs to be resized and padded.
             output_size: Pixel size of the output frame image.

        Return:
            Formatted frame with padding of specified output size.
    """

    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)

    return frame

def frames_from_video_file(video_path, n_frames, output_size = (224, 224), frame_step = 15):
    """
    Creates frames from each video file present for each category.

        Args:
            video_path: File path to the video.
            n_frames: Number of frames to be created per video file.
            output_size: Pixel size of the output frame image.

        Return:
            An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """

    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    need_length = 1 + (n_frames - 1) * frame_step

    if need_length > video_length:
        start - 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    # ret is a boolean indicating whether read was successful, frame is the image itself
    ret, frame = src.read()
    result.append(format_frames(frame, output_size))

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            # Creates an array of similar size using only 0s
            result.append(np.zeros_like(result[0]))
    src.release()
    # Re-arranges lines of code to the specificed format
    # ... means "all axes"
    result = np.array(result)[..., [2, 1, 0]]

    return result

def to_gif(images):
    """
    Creates a gif file from video frames

        Args:
            images: a file with a bunch of frames

        Returns:
            gif
    """
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animation.gif', converted_images, duration=10)
    return embed.embed_file('./animation.gif')



NUM_CLASSES = 10
FILES_PER_CLASS = 50
URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'

files = list_files_from_zip_url(URL)
files = [f for f in files if f.endswith('avi')]
files[:10]

files_for_class = get_files_per_class(files)
classes = list(files_for_class.keys())

print('# of Classes:', len(classes))
print('# of Videos for Class[0]:', len(files_for_class[classes[0]]))

files_subset = select_subset_of_classes(files_for_class, classes[:NUM_CLASSES], FILES_PER_CLASS)
print(list(files_subset.keys()))


download_dir = pathlib.Path('./UCF101_subset/')
subset_paths = download_ucf_101_subset(URL,
                                       num_classes = NUM_CLASSES,
                                       splits = {"train": 30, "val": 10, "test": 10},
                                       download_dir = download_dir)


video_count_train = len(list(download_dir.glob('train/*/*.avi')))
video_count_val = len(list(download_dir.glob('val/*/*.avi')))
video_count_test = len(list(download_dir.glob('test/*/*.avi')))
video_total = video_count_train + video_count_val + video_count_test
print(f"Total videos: {video_total}")


video_path = "End_of_a_jam.ogv"

sample_video = frames_from_video_file(video_path, n_frames=10)
sample_video.shape

ucf_sample_video = frames_from_video_file(next(subset_paths['train'].glob('*/*.avi')), 50)
to_gif(ucf_sample_video)
to_gif(sample_video)
