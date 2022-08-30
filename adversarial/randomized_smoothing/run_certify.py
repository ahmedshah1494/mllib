# evaluate a smoothed classifier on a dataset
import argparse
from importlib import import_module
from mllib.adversarial.randomized_smoothing.core import Smooth
from time import time
import torch
import torchvision
import datetime
from tqdm import trange

from mllib.datasets.dataset_factory import ImageDatasetFactory, SupportedDatasets

def get_task_class_from_str(s):
    split = s.split('.')
    modstr = '.'.join(split[:-1])
    cls_name =  split[-1]
    mod = import_module(modstr)
    task_cls = getattr(mod, cls_name)
    return task_cls

parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=SupportedDatasets._member_names_, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--task", type=str, help="task")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    base_classifier = torch.load(args.base_classifier)
    if args.task is not None:
        task = get_task_class_from_str(args.task)
        modelp = task.get_model_params()
        model = modelp.cls(modelp)


    idfp = ImageDatasetFactory.get_params()
    idfp.custom_transforms = [torchvision.transforms.ToTensor()]*2
    idfp.dataset = SupportedDatasets._member_map_[args.dataset]
    idfp.datafolder = '/home/mshah1/workhorse3/'
    idfp.max_num_test = args.max
    train_dataset, val_dataset, test_dataset, nclasses = ImageDatasetFactory.get_image_dataset(idfp)

    # create the smooothed classifier g
    smoothed_classifier = Smooth(base_classifier, nclasses, args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    if args.split == 'test':
        dataset = test_dataset
    if args.split == 'train':
        dataset = train_dataset
    for i in trange(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]

        before_time = time()
        # certify the prediction of g around x
        x = x.cuda()
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        f.write("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed))
        f.flush()

    f.close()