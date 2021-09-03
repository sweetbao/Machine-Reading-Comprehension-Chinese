import json
import csv
import argparse
import os
import sys

OPTS = None

def parse_args():
  parser = argparse.ArgumentParser('submission script to create csv submission from json format.')
  parser.add_argument('--pred_file', metavar='./output/predictions.json',
    help='Input predictions JSON file.')
  parser.add_argument('--out-file', '-o', metavar='./output/dev_submission.csv', 
    help='Write submissions to file.')
  return parser.parse_args()

def main():
    pred_file = OPTS.pred_file
    out_file = OPTS.out_file

    with open(pred_file, 'r') as f:
        data = json.load(f)
    headers = ['Id', 'Predicted']   
    with open(out_file, 'w') as w:
        f_csv = csv.writer(w)
        f_csv.writerow(headers)
        f_csv.writerows([(item, data[item]) for item in data])

if __name__ == '__main__':
  OPTS = parse_args()
  main()