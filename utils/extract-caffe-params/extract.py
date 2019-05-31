import caffe
import numpy as np
import argparse
import os

def extract_caffe_model(model, weights, output_path):
  """extract caffe model's parameters to numpy array, and write them to files
  Args:
    model: path of '.prototxt'
    weights: path of '.caffemodel'
    output_path: output path of numpy params 
  Returns:
    None
  """
  net = caffe.Net(model, caffe.TEST)
  net.copy_from(weights)
  # filename = "../../pretrained_models/casenet_iter_22000.npz"
  # filename = "../../init_models/casenet_inst_init.npz"

  if not os.path.exists(output_path):
    os.makedirs(output_path)

  kwds = {} # create an empty dictionary
  for item in net.params.items():
    name, layer = item
    print('convert layer: ' + name)

    num = 0
    for p in net.params[name]:
      # outname = str(name) + '_' + str(num)
      # kwds[outname] =  p.data
      np.save(output_path + '/' + str(name) + '_' + str(num), p.data)
      num += 1

  # add all npy files to keywords dictionary
  # for file in os.listdir("../../pretrained_models/"):
  #   if file.endswith(".npy"):
  #     print('saving layer: ' + file[:-4])
  #     arr = np.load("../../pretrained_models/" + file)
  #     file = file[:-4] # remove extension
  #     kwds[file] = arr

  # np.savez(filename, **kwds)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", help="model prototxt path .prototxt")
  parser.add_argument("--weights", help="caffe model weights path .caffemodel")
  parser.add_argument("--output", help="output path")
  args = parser.parse_args()
  extract_caffe_model(args.model, args.weights, args.output)