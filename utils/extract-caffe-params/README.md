# Extract-caffe-params
By *[nilboy](https://github.com/nilboy)*

extract caffe model's parameters to numpy array, and write them to files.

For help on usage, 
`python extract.py -h`

To use, 
```python extract.py [-h] [--model MODEL] [--weights WEIGHTS] [--output OUTPUT]

Optional Arguments:
      -h, --help         show this help message and exit
      --model MODEL      model prototxt path .prototxt
      --weights WEIGHTS  caffe model weights path .caffemodel
      --output OUTPUT    output path
```

## Example
`python extract.py --model=init.prototxt --weights=init.caffemodel --output=out`
