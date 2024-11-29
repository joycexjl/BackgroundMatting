"""
This file records the directory paths to the different datasets.
You will need to configure it for training the model.

All datasets follows the following format, where fgr and pha points to directory that contains jpg or png.
Inside the directory could be any nested formats, but fgr and pha structure must match. You can add your own
dataset to the list as long as it follows the format. 'fgr' should point to foreground images with RGB channels,
'pha' should point to alpha images with only 1 grey channel.
{
    'YOUR_DATASET': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        }
    }
}
"""

BASE_PATH = '/home/stu1/LXL/datasets/'

DATA_PATH = {
    'ppm100': {
        'train': {
            'fgr': BASE_PATH + 'PPM-100/' + 'train/image',
            'pha': BASE_PATH + 'PPM-100/' + 'train/matte'
        },
        'valid': {
            'fgr': BASE_PATH + 'PPM-100/' + 'valid/image',
            'pha': BASE_PATH + 'PPM-100/' + 'valid/matte'
        }
    },
    'videomatte240k': {
        'train': {
            'fgr': BASE_PATH + 'videomatte240k/' + 'train/fgr',
            'pha': BASE_PATH + 'videomatte240k/' + 'train/pha'
        },
        'valid': {
            'fgr': BASE_PATH + 'videomatte240k/' + 'test/fgr',
            'pha': BASE_PATH + 'videomatte240k/' + 'test/pha'
        }
    },
    'photomatte13k': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        }
    },
    'distinction': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
    },
    'adobe': {
        'train': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR',
        },
        'valid': {
            'fgr': 'PATH_TO_IMAGES_DIR',
            'pha': 'PATH_TO_IMAGES_DIR'
        },
    },
    'backgrounds': {
        'train': BASE_PATH + 'Backgrounds/train',
        'valid': BASE_PATH + 'Backgrounds/test'
    },
}
