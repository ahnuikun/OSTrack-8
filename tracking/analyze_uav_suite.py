import argparse
import os
import sys

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.analysis.plot_results import print_per_sequence_results, print_results
from lib.test.evaluation import get_dataset, trackerlist


DATASETS = ['visdrone', 'uav123', 'uavdt', 'dtb70', 'lasot']


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze OSTrack results on the UAV/LaSOT evaluation suite.')
    parser.add_argument('--tracker_name', type=str, default='ostrack', help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, required=True,
                        help='Name of tracker parameter/config file used during testing.')
    parser.add_argument('--dataset', type=str, default='all', choices=['all'] + DATASETS,
                        help='Dataset to analyze. Use "all" to analyze all configured datasets.')
    parser.add_argument('--display_name', type=str, default=None,
                        help='Display name shown in the result table. Defaults to tracker_param.')
    parser.add_argument('--force_evaluation', action='store_true',
                        help='Recompute cached evaluation data from tracking result txt files.')
    parser.add_argument('--skip_missing_seq', action='store_true',
                        help='Skip sequences without result files instead of raising an error.')
    parser.add_argument('--per_sequence', action='store_true',
                        help='Print per-sequence overlap scores after the dataset summary.')
    return parser.parse_args()


def main():
    args = parse_args()
    datasets = DATASETS if args.dataset == 'all' else [args.dataset]
    display_name = args.display_name or args.tracker_param

    for dataset_name in datasets:
        print('=' * 80)
        print('Analyzing dataset: {}'.format(dataset_name))
        print('=' * 80)

        trackers = trackerlist(name=args.tracker_name, parameter_name=args.tracker_param,
                               dataset_name=dataset_name, run_ids=None,
                               display_name=display_name)
        dataset = get_dataset(dataset_name)

        print_results(trackers, dataset, dataset_name, merge_results=True,
                      plot_types=('success', 'norm_prec', 'prec'),
                      force_evaluation=args.force_evaluation,
                      skip_missing_seq=args.skip_missing_seq)

        if args.per_sequence:
            print_per_sequence_results(trackers, dataset, dataset_name, merge_results=True,
                                       force_evaluation=args.force_evaluation,
                                       skip_missing_seq=args.skip_missing_seq)


if __name__ == '__main__':
    main()
