import glob
import numpy as np
import os
import time
from multiprocessing import Event, Lock, Process, Queue
from PIL import Image
from tqdm import tqdm

from jaccard import *


src_data_path = 'test/data/in/'
res_data_path = 'test/data/out/'
model_path = 'test/data/models/miniunet.hdf5'
margin = 2
gpu_count = 2
predict_batch_size=32


def process_image(im_path, model, input_size, output_size, margin, batch_size, log, lock):
    """
    Compute mask image using CNN model
    :return: mask image
    """
    from keras.preprocessing.image import img_to_array, array_to_img, load_img

    t = time.time()

    work_type = np.float32
    img = load_img(im_path)

    sx = img.size[0]
    sy = img.size[1]

    patch_step = (output_size[0] - 2*margin, output_size[1] - 2*margin)
    count_x = sx / patch_step[0]
    count_y = sy / patch_step[1]
    if sx % patch_step[0] > 0:
        count_x = count_x + 1
    if sy % patch_step[1] > 0:
        count_y = count_y + 1

    patches = np.zeros(shape=(count_x, input_size[0], input_size[1], 3),
                       dtype=work_type)
    labels = np.zeros(shape=(count_x, 2), dtype=np.int32)
    res_image = Image.new('L', img.size, 1)
    print( 'count_x =', count_x )
    for y in range(count_y):
        patch_n = 0
        for x in range(count_x):
            dxy = ((input_size[0] - patch_step[0])/2, (input_size[1] - patch_step[1])/2)
            patch_x = x * patch_step[0] - dxy[0]
            patch_y = y * patch_step[1] - dxy[1]
            patch = img.crop((patch_x, patch_y, patch_x + input_size[0], patch_y + input_size[1]))
            patch = img_to_array(patch)
            labels[patch_n] = np.array((patch_x + dxy[0] - margin, patch_y + dxy[1] - margin), dtype=np.int32)
            patches[patch_n] = np.array(patch, dtype=work_type)/255.0
            patch_n = patch_n + 1

        res = model.predict(patches, batch_size=batch_size, verbose=0)

        for pos, im in zip(labels, res):
            x = pos[0]
            y = pos[1]
            im = array_to_img(np.array(im*255, dtype=np.uint8), scale=False)
            box = (x, y, x + im.size[0], y + im.size[1])
            res_image.paste(im=im, box=box)

    process_time = time.time() - t
    dev_id = os.environ['CUDA_VISIBLE_DEVICES']
    with lock:
        line = '{} process {:>8.3f} s sum {:>9.0f} dev_id {}\n'
        log.write(line.format(im_path, process_time, img_to_array(res_image).sum(), dev_id))
        log.flush()
    return res_image


class MultiGpuWorker(Process):
    def __init__(self, _dev_id, _queue, _errev, _model_path, **_args):
        Process.__init__(self, name='MultiGpuWorker')
        self.dev_id = _dev_id
        self.queue = _queue
        self.errev = _errev
        self.model_path = _model_path
        self.args = _args

    def run(self):
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.dev_id)

        # print('Worker %d start with dev_id %s' % (self.pid, os.environ['CUDA_VISIBLE_DEVICES']))
        from keras.models import load_model
        try:
            model = load_model(filepath=self.model_path,
                               custom_objects=dict([('jaccard_distance', jaccard_distance),
                                                    ('jaccard_index', jaccard_index)]))
            # print('Model name %s input %s output %s' % (model.name, model.input.shape, model.output.shape))
            input_size = (int(model.input.shape[1]), int(model.input.shape[2]))
            output_size = (int(model.output.shape[1]), int(model.output.shape[2]))
            self.args['model'] = model
            self.args['output_size'] = output_size
            self.args['input_size'] = input_size

            while not self.errev.is_set():
                im_path = self.queue.get(block=True)
                if im_path is None:
                    break
                dirname, basename = os.path.split(im_path)
                res_image = process_image(im_path=im_path, **self.args)
                res_image.save(os.path.join(res_data_path, basename + '.PNG'))
            print('Worker PID %d is finished' % self.pid)
        except Exception:
            print('Worker PID %d is failed' % self.pid)
            self.errev.set()
            raise

def main():
    flist = glob.glob(src_data_path + '/*.png')
    if len(flist) == 0:
        print('Input is empty %s' % src_data_path)
        exit(-1)

    flist.sort()

    resultlog = open(res_data_path + '/res.log', 'w')
    resultlog_lock = Lock()
    queue = Queue(maxsize=gpu_count)
    errev = Event()
    workers = []
    args = dict([('margin', margin),
                 ('batch_size', predict_batch_size),
                 ('log', resultlog),
                 ('lock', resultlog_lock)])

    for dev_id in range(gpu_count):
        workers.append(MultiGpuWorker(dev_id, queue, errev, model_path, **args))
    for w in workers:
        w.start()
    for f in tqdm(flist):
        queue.put(f)

    if errev.is_set():
        print('Some workers are not done.\n')
        exit(-1)

    queue.put(None)
    for w in workers:
        queue.put(None)
        w.join()

    resultlog.close()

if __name__ == '__main__':
    main()
